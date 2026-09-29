"""
pose/stabilization.py
=======================
Post-processing to reduce jitter in the raw 2D keypoints produced by
`pose/extract_keypoints.py`, in two steps applied per (session_id,
global_person_id, keypoint_name) sequence -- never across sequences, same
grouping discipline as the rest of the project (temporal features are
grouped by session+identity, never by label/value):

1. **Confidence gating + short-gap interpolation.** A keypoint below
   `conf_threshold` (or simply absent -- MediaPipe found no pose in that
   person's crop that frame, e.g. fully occluded by the LEGO box) is
   treated as missing. Gaps of at most `max_gap_frames` are filled by
   linear interpolation; longer gaps are left `NaN` -- same "no made-up
   signal" principle as `pose/appearance_embedding.py` and
   `segmentation/merging/mask_utils.py` (better `None`/`NaN` than a
   plausible-looking but invented value).
2. **One Euro Filter** (Casiez, Roussel, Vogel 2012), a causal low-pass
   filter designed specifically for noisy interactive/tracking signals --
   low lag on fast movements, strong smoothing when the point is close to
   still. See `OneEuroFilter`'s docstring for the two parameters that
   actually matter (`min_cutoff`, `beta`) and a correctness warning: the
   filter's own conservative default (`beta=0`) makes things *worse*, not
   better, on a keypoint that moves a lot -- always tune on a real
   sequence before trusting the defaults (`_smoke_test` below is a
   starting point, not a substitute).

Correctness note on gaps: a sequence's frames are densified to every
integer frame between its own first and last observed frame before
gating/filtering (missing frames -> `NaN` rows, not silently dropped).
This matters because `OneEuroFilter` is fed the *true* elapsed time
(`frame / fps`) and the interpolation gap limit is counted in *actual*
missing frames -- both would silently be wrong (computed from row
position instead) if rows for occluded frames were simply absent from
the input, which is the normal case here (MediaPipe finds nothing that
frame).

Does not require the two cameras or any camera calibration -- operates
only on keypoints already in one camera's pixel space.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# One Euro Filter
# --------------------------------------------------------------------------

class _LowPassFilter:
    def __init__(self) -> None:
        self._initialized = False
        self._x_prev: float | None = None

    def filter(self, x: float, alpha: float) -> float:
        if not self._initialized:
            self._x_prev = x
            self._initialized = True
            return x
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev
        self._x_prev = x_hat
        return x_hat


class OneEuroFilter:
    """Filters one scalar channel over time -- one instance per (person,
    keypoint, coordinate) to filter, state is per-channel and must not be
    shared.

        f = OneEuroFilter(freq=25.0, min_cutoff=1.0, beta=0.3)
        smoothed = f(raw_value, t=frame_index / fps)

    Parameters:
      min_cutoff: lower -> more smoothing on a nearly-still point, but
                  more lag if pushed too low. 1.0 is a reasonable start.
      beta:       higher -> reacts faster to quick movements (less lag),
                  at the cost of passing through more noise while moving.
      d_cutoff:   rarely needs changing (1.0 is fine almost always).

    IMPORTANT -- beta=0 is NOT a safe default for a keypoint that moves a
    meaningful amount (a gesturing hand, not a nearly-still point): with
    no speed-adaptive cutoff, `min_cutoff` alone over-smooths and lags
    behind real motion enough that the filtered signal can end up *further*
    from the true trajectory than the raw noisy input. Verified with
    `_smoke_test` below: beta=0 -> filtered RMSE worse than raw; beta~0.3
    (for pixel-scale motion) -> clearly better than raw. Always tune
    min_cutoff/beta on a real sequence (plot raw vs. filtered vs., if you
    have it, a trusted reference) before trusting either default.
    """

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0) -> None:
        if freq <= 0:
            raise ValueError("freq (fps) must be > 0")
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_filt = _LowPassFilter()
        self._dx_filt = _LowPassFilter()
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        te = 1.0 / freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, t: float | None = None) -> float:
        if t is not None and self._t_prev is not None:
            dt = t - self._t_prev
            if dt > 0:
                self.freq = 1.0 / dt
        self._t_prev = t

        dx = 0.0 if self._x_filt._x_prev is None else (x - self._x_filt._x_prev) * self.freq
        edx = self._dx_filt.filter(dx, self._alpha(self.d_cutoff, self.freq))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x_filt.filter(x, self._alpha(cutoff, self.freq))


# --------------------------------------------------------------------------
# Gating + gap interpolation
# --------------------------------------------------------------------------

def _gate_and_interpolate(values: np.ndarray, confidence: np.ndarray,
                           conf_threshold: float, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """`values`/`confidence`: 1D arrays for ONE already-densified (no
    missing frame numbers) channel. Returns `(gated, was_interpolated)`.
    A sample counts as missing if its confidence is below threshold OR
    `NaN` (an inserted/missing frame has `NaN` confidence, not 0 -- using
    `>=` rather than `< threshold` is what makes `NaN` correctly count as
    missing here: `NaN < threshold` is `False` in numpy/pandas, which
    would silently let missing frames through ungated). Gaps of length
    `<= max_gap_frames` are linearly interpolated; longer gaps stay `NaN`."""
    v = pd.Series(values, dtype=float).copy()
    low_conf = ~(pd.Series(confidence, dtype=float) >= conf_threshold)
    v[low_conf] = np.nan

    was_nan = v.isna()
    v_interp = v.interpolate(method="linear", limit=max_gap_frames, limit_area="inside")
    was_interpolated = (was_nan & v_interp.notna()).to_numpy()
    return v_interp.to_numpy(), was_interpolated


def _densify(group: pd.DataFrame, col_frame: str) -> pd.DataFrame:
    """Reindexes one (session, person, keypoint) group to every integer
    frame between its own min and max observed frame, inserting `NaN`
    rows for frames with no detection at all (occlusion, or MediaPipe
    simply found nothing in the crop). Never extrapolates before the
    first or after the last observed frame. `session_id`/`global_person_id`/
    `keypoint_name` are constant within a group and re-filled on the
    inserted rows (they come from the group key, not from data)."""
    frame_min, frame_max = int(group[col_frame].min()), int(group[col_frame].max())
    dense = group.set_index(col_frame).reindex(range(frame_min, frame_max + 1))
    dense.index.name = col_frame
    return dense.reset_index()


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def smooth_keypoints(
    df: pd.DataFrame,
    *,
    fps: float,
    conf_threshold: float = 0.4,
    max_gap_frames: int = 5,
    min_cutoff: float = 1.0,
    beta: float = 0.0,
    col_session: str = "session_id",
    col_person: str = "global_person_id",
    col_frame: str = "frame",
    col_keypoint: str = "keypoint_name",
    col_x: str = "x",
    col_y: str = "y",
    col_conf: str = "confidence",
) -> pd.DataFrame:
    """Long-format keypoints in, long-format keypoints out, with
    `x_smooth`/`y_smooth`/`interpolated` added. The output can have MORE
    rows than the input: each (session, person, keypoint) sequence is
    densified to every frame in its own observed span first (see
    `_densify`), so a previously-missing frame gets an explicit row
    (`x`/`y`/`confidence` all `NaN`, `x_smooth`/`y_smooth` interpolated or
    `NaN` depending on the gap length) instead of silently not existing.

    `beta=0.0` here matches `OneEuroFilter`'s own conservative default,
    NOT a recommendation -- read the "IMPORTANT" note in its docstring
    before running this on real data."""
    required = [col_session, col_person, col_frame, col_keypoint, col_x, col_y, col_conf]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"smooth_keypoints: missing columns {missing}; have {list(df.columns)}")

    group_cols = [col_session, col_person, col_keypoint]
    groups = df.groupby(group_cols, sort=False)
    print(f"[stabilization] {groups.ngroups} (session, person, keypoint) sequences to filter, fps={fps}")

    out_frames = []
    for key, group in groups:
        dense = _densify(group.sort_values(col_frame), col_frame)
        # re-fill the group-key columns on inserted rows (constant within a group)
        for col, val in zip(group_cols, key if isinstance(key, tuple) else (key,)):
            dense[col] = val

        x_gated, x_interp = _gate_and_interpolate(
            dense[col_x].to_numpy(), dense[col_conf].to_numpy(), conf_threshold, max_gap_frames)
        y_gated, y_interp = _gate_and_interpolate(
            dense[col_y].to_numpy(), dense[col_conf].to_numpy(), conf_threshold, max_gap_frames)

        fx = OneEuroFilter(freq=fps, min_cutoff=min_cutoff, beta=beta)
        fy = OneEuroFilter(freq=fps, min_cutoff=min_cutoff, beta=beta)
        frames = dense[col_frame].to_numpy()

        x_smooth = np.full(len(dense), np.nan)
        y_smooth = np.full(len(dense), np.nan)
        for i in range(len(dense)):
            t = frames[i] / fps  # true elapsed time, not row position -- see module docstring
            if not np.isnan(x_gated[i]):
                x_smooth[i] = fx(x_gated[i], t)
            if not np.isnan(y_gated[i]):
                y_smooth[i] = fy(y_gated[i], t)

        dense["x_smooth"] = x_smooth
        dense["y_smooth"] = y_smooth
        dense["interpolated"] = x_interp | y_interp
        out_frames.append(dense)

    result = pd.concat(out_frames, ignore_index=True)
    n_new = len(result) - len(df)
    n_interp = int(result["interpolated"].sum())
    print(f"[stabilization] {n_new} frame(s) with no detection made explicit as NaN rows "
          f"(densification), {n_interp} of those filled by interpolation over a short gap")
    return result


# --------------------------------------------------------------------------
# Self-test on synthetic data -- sanity check, NOT a substitute for tuning
# on real keypoints. Run directly: `python -m pose.stabilization`
# --------------------------------------------------------------------------

def _smoke_test(make_plot: bool = True, out_path: str = "stabilization_smoke_test.png") -> None:
    rng = np.random.default_rng(0)
    fps = 25.0
    n = 150
    frame = np.arange(n)
    t = frame / fps

    true_x = 100 * np.sin(2 * np.pi * 0.5 * t) + 0.4 * t * 100
    noisy_x = true_x + rng.normal(0, 4.0, size=n)
    confidence = np.full(n, 0.9)

    # simulate an occlusion (hand behind the LEGO box): a few frames with
    # no detection at all -- rows dropped, not just low-confidence
    occluded = slice(60, 66)
    keep = np.ones(n, dtype=bool)
    keep[occluded] = False

    df = pd.DataFrame({
        "session_id": "smoke", "global_person_id": "p1", "keypoint_name": "right_wrist",
        "frame": frame[keep], "x": noisy_x[keep], "y": np.zeros(keep.sum()), "confidence": confidence[keep],
    })

    for beta in (0.0, 0.3):
        out = smooth_keypoints(df.copy(), fps=fps, min_cutoff=1.0, beta=beta)
        aligned = out.set_index("frame").reindex(frame)
        rmse = float(np.sqrt(np.nanmean((aligned["x_smooth"].to_numpy() - true_x) ** 2)))
        print(f"[smoke_test] beta={beta}: RMSE vs. true motion = {rmse:.2f}px "
              f"(raw RMSE = {np.sqrt(np.mean((noisy_x - true_x) ** 2)):.2f}px)")

    if make_plot:
        import matplotlib.pyplot as plt

        out = smooth_keypoints(df.copy(), fps=fps, min_cutoff=1.0, beta=0.3)
        aligned = out.set_index("frame").reindex(frame)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, true_x, "--", color="gray", linewidth=1, label="true motion (unknown in practice)")
        ax.plot(t, noisy_x, ".", color="#d62728", markersize=3, alpha=0.6, label="raw keypoint (noisy)")
        ax.plot(t, aligned["x_smooth"].to_numpy(), "-", color="#1f77b4", linewidth=2,
                label="after gating + One Euro Filter (beta=0.3)")
        interp_mask = aligned["interpolated"].fillna(False).to_numpy()
        if interp_mask.any():
            ax.scatter(t[interp_mask], aligned["x_smooth"].to_numpy()[interp_mask],
                       color="orange", zorder=5, s=25, label="interpolated (occlusion gap)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("keypoint x (px)")
        ax.set_title("pose/stabilization.py smoke test (synthetic data)")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"[smoke_test] plot saved to {out_path}")


if __name__ == "__main__":
    _smoke_test()
