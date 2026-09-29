"""
pose/movement_metrics.py
==========================
First-pass movement metrics (distance traveled, velocity, variability)
from a `pose/depth.py` 3D keypoints CSV -- the actual analytical payoff
behind extracting depth at all (see `pose/depth.py`'s module docstring
and README).

Representative "root" point per person per frame
------------------------------------------------------
COCO-17 has no pelvis/root joint, so this module uses the midpoint of
`left_hip`/`right_hip` as a stand-in (falls back to whichever single
hip is valid if only one is, `NaN` if neither is -- no invented
substitute). A reasonable proxy for whole-body position, not a precise
center of mass.

What "no made-up signal" means here specifically
------------------------------------------------------
Distance traveled is the sum of frame-to-frame 3D displacements, but
ONLY between two frames that both have a valid root position -- a gap
(missing pose, missing depth, or both) contributes NOTHING to the
total, rather than assuming a straight-line path across the gap. This
UNDER-counts true distance whenever there's a gap (a real limitation,
not hidden -- see `coverage_fraction` in the summary), but never
invents movement that was never actually observed. `velocity_mm_s` at
a given frame is that frame's displacement divided by the ACTUAL
elapsed time since the last valid frame (`dt_s`, which can be more than
one frame's worth across a gap), not divided by a fixed `1/fps`.

Smoothing the root's 3D trajectory (unfiltered depth noise)
------------------------------------------------------------
`x`/`y` PIXEL coordinates are already smoothed in `stabilization.py`
before the depth lookup, but the pixel-to-mm backprojection then
multiplies those (already-smooth) pixel coordinates by `Z` (depth),
which is NOT smoothed -- so raw sensor jitter in Z propagates into
`X_mm`/`Y_mm` too, not just `Z_mm`. A displacement is always >= 0, so
this jitter never cancels out when summed frame-to-frame -- it only
ever inflates `total_distance_mm`.

Confirmed on a real session (`10_individual_12`, 2026-09-22): a person
who spent the session mostly seated at the table accumulated
`total_distance_mm` implying an average speed above 1 m/s sustained
continuously for ~14 minutes -- not physically plausible for this kind
of session (walking pace, non-stop, for the entire clip). The
per-frame raw displacement averaged ~69mm/frame at 15 fps, well under
the `max_velocity_mm_s` outlier ceiling below, so none of it was ever
rejected as an outlier -- it's persistent moderate noise, not rare
spikes.

`smooth_root` (default on) passes the root's X/Y/Z trajectory through a
`OneEuroFilter` per axis -- the SAME filter class already used in
`stabilization.py` for x/y pixels, just applied downstream to the
backprojected mm position instead. `root_min_cutoff`/`root_beta` are
tuned for mm/s-scale speeds (NOT the same numeric defaults as
`stabilization.py`'s pixel-scale ones). Chosen via a synthetic sweep
(`tune_root_smoothing.py`, kept alongside the tests) calibrated so the
synthetic near-still noise reproduces the real ~69mm/frame average
seen on `10_individual_12`: `min_cutoff=0.6, beta=0.0` suppresses
~85% of that stationary noise (900 synthetic frames) while recovering
~73-91% of a synthetic sustained 1500mm/s walking burst's true
distance, depending on how long the burst lasts (1s vs. 3s -- a SHORT
real movement is recovered less completely than a long one, an
inherent property of any fixed low-pass filter, not a bug).

`beta` was deliberately left at 0 (non-adaptive): a nonzero beta was
tried and made BOTH suppression and recovery numbers look better in
isolation, but only because noise itself produces an apparent "speed"
that the adaptive term reacts to -- even `beta=0.05` collapsed
stationary-noise suppression from ~94% to ~31% in testing, i.e. it
mostly stopped filtering. `stabilization.py`'s pixel-domain use case
does not have this problem as severely because its noise floor is
much smaller relative to real motion.

Known limitations -- read before trusting `total_distance_mm` at face
value:
- A fixed low-pass filter fundamentally trades noise suppression
  against response lag -- there is NO single `min_cutoff` that fully
  solves both a stationary person's jitter AND a brief (under ~1s)
  real movement burst. `root_min_cutoff`/`root_beta` are exposed on
  the CLI specifically so this can be re-tuned per use case.
- A residual noise floor remains, and it SCALES WITH SESSION LENGTH:
  smoothing reduces the per-frame noise contribution sharply but does
  not eliminate it, and a displacement is always >= 0, so even a
  small residual never cancels out over thousands of frames. On a
  synthetic ~14-minute session shaped like a real one (mostly still +
  a few walking episodes), the smoothed `total_distance_mm` was still
  several times the TRUE distance covered by the walking episodes
  alone -- an enormous improvement over the ~50x inflation seen
  unsmoothed, but not "true meters walked". Treat `total_distance_mm`
  on a full session as most trustworthy for RELATIVE comparisons
  (across people, sessions, or before/after) rather than as an
  absolute physical distance, especially across sessions of different
  length or `coverage_fraction`.

Rejecting single-frame velocity spikes (unfiltered Z noise)
------------------------------------------------------------
`max_velocity_mm_s` is a SEPARATE, complementary safety net -- it
still runs after smoothing, on the smoothed trajectory. Confirmed on
the same real session that a handful of frames (well under 2%) had a
`velocity_mm_s` of several METERS per second, tens of thousands of
mm/s at the extreme -- physically impossible for a person, a single
bad depth reading producing a one-frame "teleport" that smoothing
alone (a low-pass filter) does not fully absorb. Left uncorrected,
that handful of frames distorted `total_distance_mm`/
`mean_velocity_mm_s` by 15-37% on that same session -- not a rounding
error.

`max_velocity_mm_s` filters this: any single-frame displacement that
would imply a velocity above this ceiling is treated as an unreliable
READING (not a real position), exactly like an `x`/`y` gap -- it gets
`NaN` for that frame's `displacement_mm`/`velocity_mm_s`, and, just as
important, that frame's position is NOT used as the anchor for the
NEXT displacement either (so one bad frame corrupts at most one
skipped step, not two). The default (4000 mm/s = 4 m/s, a fast jog) is
deliberately generous: confirmed on the same real session that a
genuine brief standing-and-walking episode produces a SUSTAINED
elevated velocity across many consecutive frames in the 800-2500 mm/s
range, not an isolated spike -- the two patterns look different in the
data (sustained vs. one-frame), and 4000 mm/s sits comfortably above
real walking speed while still catching the tens-of-thousands-mm/s
sensor artifacts. `n_outliers_rejected`/`outlier_fraction` in the
summary report how many frames this affected -- a high fraction is
itself worth investigating (would suggest a session-wide depth-quality
problem, not occasional noise), never silently discarded.

Suppressing the residual sub-threshold noise floor (deadzone)
------------------------------------------------------------
`root_min_cutoff`/`root_beta` smoothing (above) suppresses most, not
all, of the stationary noise, and `max_velocity_mm_s` only catches
rare one-frame teleports -- neither one addresses the residual,
smoothed-but-still-nonzero jitter that a seated person keeps producing
on EVERY frame, which never cancels out (displacement >= 0) and keeps
accumulating over a long session (see "Known limitations" above: on a
synthetic 14-minute stationary session shaped like the real
`10_individual_12` noise level, that residual alone summed to
>130 METERS of fake "distance travelled" even after smoothing).

`still_deadzone_mm_s` is a second, complementary filter for exactly
this: any SMOOTHED frame-to-frame displacement implying a velocity at
or below this is treated as "no observed movement" rather than a real
(tiny) step -- `displacement_mm`/`velocity_mm_s` are recorded as `0.0`
(not `NaN`: this is a real, positive observation of stillness, not a
missing/rejected reading), and `is_deadzone` is set.

This is deliberately a PER-FRAME check against the immediately
preceding usable frame, exactly like the rest of the displacement
chain -- a below-threshold frame's position still becomes the anchor
for the NEXT comparison (unlike `max_velocity_mm_s` rejection, whose
whole point is that a bad reading must not anchor anything). An
earlier version of this held the anchor fixed across an entire
below-threshold run (the same discipline as outlier rejection), which
looked ideal on synthetic stationary noise (0mm of fake distance over
a 14-minute synthetic stationary session, vs. the >130m left over from
smoothing alone) -- but real walking is not a straight line away from
one fixed point: a person paces, turns, and returns near where they
started. Confirmed on a synthetic back-and-forth walk (~18m true path
over 30s): holding the anchor made the measured distance collapse to
~1.4m (7.7% of the true path) because the path kept curving back near
the frozen anchor before ever clearing the threshold -- i.e. it could
silently erase real, sustained walking, which is a worse failure than
leaving some noise in. Reported directly by Michele testing the real
overlay: walking around the room still read as 0 speed. Switched to
the per-frame check for that reason: it recovers ~78% of a real
back-and-forth walk's already-smoothed distance (vs. anchor-hold's
7.7%) and leaves mover/seated separation solidly intact (>40x in the
same synthetic mover-vs-seated check used for the smoothing tuning),
at the cost of a real but much smaller residual than smoothing alone:
~13.6m of fake distance over the same synthetic 14-minute stationary
session (vs. >130m with no deadzone at all, and 0m -- but at the cost
of erasing real movement -- with the old anchor-hold design). This
residual exists because a fixed velocity threshold alone cannot
distinguish "this one frame's noise happened to exceed the threshold"
from "this one frame is part of real motion" without looking at
neighboring frames -- a possible future refinement, not attempted here
to keep this fix simple and directly re-testable against the same
regression Michele hit.

The default (270.9 mm/s) is CALIBRATED, not guessed, the same way
`root_min_cutoff`/`root_beta` are: it's the 95th percentile of
per-frame velocity in a synthetic near-stationary trajectory whose
noise sigma is chosen so its RAW displacement reproduces the real
~69.4mm/frame average measured on `10_individual_12`, after applying
this module's own default smoothing (`min_cutoff=0.6, beta=0.0`) --
i.e. it's set from the residual noise that's actually left over AFTER
smoothing, not the raw noise or a round number. A HIGHER threshold
trades more of that stationary residual away for proportionally MORE
of a real slow/curvy movement's distance lost too (confirmed: raising
it to 450mm/s zeroes the synthetic stationary residual entirely, but
also cuts the back-and-forth walk's recovered distance to ~13% of true
-- worse, not better, for telling movement apart from stillness). Like
`max_velocity_mm_s`, this is a per-session-tarable heuristic, not a
universal constant -- it would need re-tuning for a different sensor,
distance-to-subject, or frame rate, which is why it's exposed on the
CLI rather than hardcoded. `n_deadzone_frames`/`deadzone_fraction` in
the summary report how many frames this affected, same transparency
as `outlier_fraction` -- never silently discarded. Pass `None` (or 0)
to disable.

`segmentation/tools/overlay_movement.py`'s own `still_deadzone_kmh`
display option defaults to this SAME calibrated value (converted to
km/h) for consistency -- see that module's docstring. When a
`movement_per_frame.csv` was already computed with this deadzone on
(the default), the overlay's display-time zeroing is mostly a no-op;
it stays useful for CSVs computed with the deadzone off/overridden, or
to preview a different threshold without recomputing the metrics.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pose.stabilization import OneEuroFilter

_ROOT_KEYPOINTS = ("left_hip", "right_hip")

# Tuned on synthetic data (see tune_root_smoothing.py), NOT pixel-scale
# defaults -- speeds here are in mm/s (hundreds to low thousands for real
# movement), a very different regime from stabilization.py's pixel/s scale.
_DEFAULT_ROOT_MIN_CUTOFF = 0.6
_DEFAULT_ROOT_BETA = 0.0

# 95th percentile of post-smoothing (min_cutoff=0.6, beta=0.0) per-frame
# velocity on a synthetic near-stationary trajectory calibrated to reproduce
# the real ~69.4mm/frame raw noise average measured on 10_individual_12 (see
# module docstring's "Suppressing the residual sub-threshold noise floor" and
# test_root_smoothing.py) -- a per-session-tarable heuristic, not a universal
# constant, same spirit as max_velocity_mm_s's default.
DEFAULT_STILL_DEADZONE_MM_S = 270.9


def _root_position(df_3d: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, person, frame) with a root X/Y/Z -- mean of
    `left_hip`/`right_hip`'s valid values (see module docstring). NaN
    values and altogether-missing keypoint rows both correctly reduce
    to "use whichever is valid" / "NaN if neither is", since
    `groupby.mean()` skips NaN by default and a keypoint absent from
    `df_3d` for that frame simply isn't in the group to begin with."""
    hips = df_3d[df_3d["keypoint_name"].isin(_ROOT_KEYPOINTS)]
    return (hips.groupby(["session_id", "global_person_id", "frame"])[["X_mm", "Y_mm", "Z_mm"]]
                .mean()
                .reset_index())


def _detect_raw_outliers(frame: np.ndarray, pos_raw: np.ndarray, valid: np.ndarray, *,
                          fps: float, max_velocity_mm_s: float | None) -> np.ndarray:
    """Flags single-frame RAW jumps that imply a velocity above
    `max_velocity_mm_s` -- exactly the original (pre-smoothing) outlier
    logic, run FIRST on the unsmoothed position, deliberately BEFORE
    any smoothing touches the data. See `compute_movement_metrics` for
    why the ordering matters: if smoothing ran first and outlier
    detection second, a single bad raw reading would still leak into
    the filter's internal recursive state (and therefore into several
    SUBSEQUENT frames' smoothed positions) even though that one frame
    gets correctly rejected -- silently breaking the documented "one
    bad frame corrupts at most one skipped step" guarantee. Detecting
    on raw data first means a flagged frame is never fed to the filter
    at all, exactly like a missing/invalid frame."""
    is_outlier = np.zeros(len(pos_raw), dtype=bool)
    if max_velocity_mm_s is None:
        return is_outlier
    last_ok_idx = None
    for i in range(len(pos_raw)):
        if not valid[i]:
            continue
        if last_ok_idx is not None:
            d = float(np.linalg.norm(pos_raw[i] - pos_raw[last_ok_idx]))
            dt = (frame[i] - frame[last_ok_idx]) / fps
            v = d / dt if dt > 0 else np.nan
            if not np.isnan(v) and v > max_velocity_mm_s:
                is_outlier[i] = True
                continue  # last_ok_idx stays put -- this frame anchors nothing
        last_ok_idx = i
    return is_outlier


def _smooth_root_xyz(frame: np.ndarray, pos: np.ndarray, usable: np.ndarray, *,
                      fps: float, min_cutoff: float, beta: float) -> np.ndarray:
    """Applies one `OneEuroFilter` per axis (X, Y, Z) to this person's
    USABLE root positions, in frame order -- see module docstring's
    "Smoothing the root's 3D trajectory" for why this is needed even
    though the pixel x/y this was backprojected from is already smooth.

    `usable` must already exclude both invalid (NaN) frames AND raw
    outlier frames (see `_detect_raw_outliers`) -- a frame outside
    `usable` is never fed to the filter, exactly like a gap, so a bad
    raw reading cannot contaminate the filter's recursive state for
    later frames. Each filter instance is fresh per person/axis (state
    must not be shared, same discipline as
    `stabilization.smooth_keypoints`) and gets the TRUE elapsed time
    (`frame / fps`) at each usable sample, so a gap naturally slows the
    filter's own re-adaptation instead of treating a post-gap sample as
    if it were the very next frame."""
    smoothed = pos.copy()
    idx_usable = np.flatnonzero(usable)
    if idx_usable.size == 0:
        return smoothed
    filters = [OneEuroFilter(freq=fps, min_cutoff=min_cutoff, beta=beta) for _ in range(3)]
    for i in idx_usable:
        t = float(frame[i]) / fps
        for axis in range(3):
            smoothed[i, axis] = filters[axis](float(pos[i, axis]), t)
    return smoothed


def compute_movement_metrics(
    df_3d: pd.DataFrame, *, fps: float, max_velocity_mm_s: float | None = 4000.0,
    smooth_root: bool = True,
    root_min_cutoff: float = _DEFAULT_ROOT_MIN_CUTOFF,
    root_beta: float = _DEFAULT_ROOT_BETA,
    still_deadzone_mm_s: float | None = DEFAULT_STILL_DEADZONE_MM_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns `(per_frame, summary)`:

    `per_frame`: one row per (session, person, frame) with root X/Y/Z
    (mm) -- the SMOOTHED position when `smooth_root` is True (default;
    the raw, unsmoothed backprojected position is kept alongside as
    `X_mm_raw`/`Y_mm_raw`/`Z_mm_raw` for comparison, nothing is
    hidden), `displacement_mm` (3D distance from the last CONFIRMED
    frame for this person -- NaN for the first valid frame or any
    frame with no valid root itself; `0.0`, not NaN, for a frame
    rejected as sub-threshold noise -- see `is_deadzone`), `dt_s`,
    `velocity_mm_s`, `is_outlier` (True where this frame's own
    position was rejected as an unreliable reading -- see module
    docstring's "Rejecting single-frame velocity spikes"), and
    `is_deadzone` (True where a real displacement was measured but
    treated as noise and zeroed -- see "Suppressing the residual
    sub-threshold noise floor"). A frame is never both.

    `summary`: one row per (session, person) with `total_distance_mm`,
    `mean_velocity_mm_s`, `velocity_std_mm_s` (the "movement
    variability" figure), `frames_with_root`, `n_outliers_rejected`,
    `outlier_fraction`, `n_deadzone_frames`, `deadzone_fraction`,
    `total_frames`, and `coverage_fraction` -- always report
    `coverage_fraction` (and `outlier_fraction`/`deadzone_fraction`)
    alongside the distance/velocity numbers: a low `coverage_fraction`
    or a high `outlier_fraction`/`deadzone_fraction` means those
    numbers are based on a small, possibly unrepresentative, or
    noisier-than-usual (or more stationary-than-usual) slice of the
    session (see module docstring).

    `smooth_root`/`root_min_cutoff`/`root_beta`: see module docstring's
    "Smoothing the root's 3D trajectory". Pass `smooth_root=False` to
    get the old (pre-smoothing) behaviour exactly, e.g. for comparison.

    `max_velocity_mm_s`: single-frame jumps implying a velocity above
    this are treated as an unreliable reading and rejected (`None`
    disables this -- see module docstring for why 4000 mm/s is a
    reasonable default here, not a universal constant). ALWAYS runs on
    the RAW trajectory first, before any smoothing -- see
    `_detect_raw_outliers`'s docstring for why the ordering matters (a
    raw outlier must never be allowed to contaminate the smoothing
    filter's internal state).

    `still_deadzone_mm_s`: smoothed frame-to-frame displacements
    implying a velocity at or below this are zeroed instead of
    accumulating into `total_distance_mm` (`None` or `0` disables this
    -- see module docstring's "Suppressing the residual sub-threshold
    noise floor" for why `smooth_root` alone is not enough and why
    270.9 mm/s is a calibrated, per-session-tarable default, not a
    universal constant). Runs AFTER smoothing, on the smoothed
    trajectory (unlike `max_velocity_mm_s`, which runs before)."""
    root = _root_position(df_3d)
    span_by_person = df_3d.groupby(["session_id", "global_person_id"])["frame"].agg(
        lambda s: int(s.max() - s.min() + 1))

    out_frames = []
    summaries = []
    for (session_id, pid), g in root.groupby(["session_id", "global_person_id"], sort=False):
        g = g.sort_values("frame").reset_index(drop=True)
        pos_raw = g[["X_mm", "Y_mm", "Z_mm"]].to_numpy()
        valid = ~np.isnan(pos_raw).any(axis=1)
        frame = g["frame"].to_numpy()

        # Phase 1: outlier detection ALWAYS on raw data, regardless of smooth_root
        # -- see _detect_raw_outliers docstring for why this must come first.
        is_outlier = _detect_raw_outliers(frame, pos_raw, valid, fps=fps,
                                           max_velocity_mm_s=max_velocity_mm_s)
        usable = valid & ~is_outlier

        # Phase 2: smoothing sees only usable (valid, non-outlier) raw frames --
        # an outlier is skipped exactly like a gap, never fed to the filter.
        if smooth_root:
            pos = _smooth_root_xyz(frame, pos_raw, usable, fps=fps,
                                    min_cutoff=root_min_cutoff, beta=root_beta)
        else:
            pos = pos_raw

        # Phase 3: displacement/velocity, chaining consecutive USABLE frames only
        # (an outlier frame anchors nothing, same guarantee as before: one bad
        # raw frame corrupts at most one skipped step, never a cascade).
        #
        # still_deadzone_mm_s is checked PER FRAME against the immediately
        # preceding usable frame -- a below-threshold frame still becomes the
        # reference for the next comparison, it just doesn't count towards
        # displacement/velocity itself. Deliberately NOT anchor-held (unlike
        # outlier rejection): see module docstring's "Suppressing the
        # residual sub-threshold noise floor" for why holding the anchor
        # across a below-threshold run looked good on synthetic stationary
        # noise but silently erased real non-monotonic movement (confirmed
        # both synthetically and by Michele testing the real overlay).
        displacement = np.full(len(g), np.nan)
        dt_s = np.full(len(g), np.nan)
        velocity = np.full(len(g), np.nan)
        in_deadzone = np.zeros(len(g), dtype=bool)

        last_usable_idx = None
        for i in range(len(g)):
            if not usable[i]:
                continue
            if last_usable_idx is not None:
                d = float(np.linalg.norm(pos[i] - pos[last_usable_idx]))
                dt = (frame[i] - frame[last_usable_idx]) / fps
                v = d / dt if dt > 0 else np.nan
                if (still_deadzone_mm_s and not np.isnan(v)
                        and v <= still_deadzone_mm_s):
                    displacement[i] = 0.0
                    dt_s[i] = dt
                    velocity[i] = 0.0
                    in_deadzone[i] = True
                else:
                    displacement[i] = d
                    dt_s[i] = dt
                    velocity[i] = v
            last_usable_idx = i  # advances every usable frame regardless -- see above
        n_outliers = int(is_outlier.sum())
        n_deadzone = int(in_deadzone.sum())

        g = g.copy()
        if smooth_root:
            g["X_mm_raw"] = pos_raw[:, 0]
            g["Y_mm_raw"] = pos_raw[:, 1]
            g["Z_mm_raw"] = pos_raw[:, 2]
            g["X_mm"] = pos[:, 0]
            g["Y_mm"] = pos[:, 1]
            g["Z_mm"] = pos[:, 2]
        g["displacement_mm"] = displacement
        g["dt_s"] = dt_s
        g["velocity_mm_s"] = velocity
        g["is_outlier"] = is_outlier
        g["is_deadzone"] = in_deadzone
        out_frames.append(g)

        n_valid = int(valid.sum())
        n_used = n_valid - n_outliers  # valid positions actually anchoring the chain
        total_frames = int(span_by_person.get((session_id, pid), n_valid))
        summaries.append({
            "session_id": session_id,
            "global_person_id": pid,
            "total_distance_mm": float(np.nansum(displacement)),
            "mean_velocity_mm_s": float(np.nanmean(velocity)) if n_used > 1 else np.nan,
            "velocity_std_mm_s": float(np.nanstd(velocity)) if n_used > 1 else np.nan,
            "frames_with_root": n_valid,
            "n_outliers_rejected": n_outliers,
            "outlier_fraction": (n_outliers / n_valid) if n_valid else np.nan,
            "n_deadzone_frames": n_deadzone,
            "deadzone_fraction": (n_deadzone / n_valid) if n_valid else np.nan,
            "total_frames": total_frames,
            "coverage_fraction": (n_valid / total_frames) if total_frames else np.nan,
        })

    per_frame = pd.concat(out_frames, ignore_index=True) if out_frames else root
    summary = pd.DataFrame(summaries)
    return per_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computes distance traveled / velocity / movement variability per person from a "
                     "pose/depth.py 3D keypoints CSV -- see module docstring.")
    parser.add_argument("--keypoints-3d-csv", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-per-frame", default=None, help="Optional: also write the per-frame velocity time series CSV")
    parser.add_argument("--max-velocity-mm-s", type=float, default=4000.0,
                         help="Reject single-frame jumps implying a velocity above this (mm/s) as sensor "
                              "noise, not real movement -- see module docstring. Pass a very large value "
                              "(e.g. 1e12) to effectively disable this.")
    parser.add_argument("--no-smooth-root", action="store_true",
                         help="Disable the root X/Y/Z OneEuroFilter smoothing (see module docstring) and "
                              "use raw backprojected positions, i.e. the old pre-smoothing behaviour.")
    parser.add_argument("--root-min-cutoff", type=float, default=_DEFAULT_ROOT_MIN_CUTOFF,
                         help="OneEuroFilter min_cutoff for root X/Y/Z smoothing (mm/s-scale, NOT the same "
                              "numeric default as stabilization.py's pixel-scale one).")
    parser.add_argument("--root-beta", type=float, default=_DEFAULT_ROOT_BETA,
                         help="OneEuroFilter beta for root X/Y/Z smoothing (mm/s-scale).")
    parser.add_argument("--still-deadzone-mm-s", type=float, default=DEFAULT_STILL_DEADZONE_MM_S,
                         help="Smoothed frame-to-frame displacements implying a velocity at or below "
                              "this (mm/s) are zeroed instead of accumulating into total_distance_mm -- "
                              "see module docstring for why root smoothing alone isn't enough and why "
                              "this default is calibrated, not a universal constant. Pass 0 to disable.")
    args = parser.parse_args()

    df_3d = pd.read_csv(args.keypoints_3d_csv)
    per_frame, summary = compute_movement_metrics(
        df_3d, fps=args.fps, max_velocity_mm_s=args.max_velocity_mm_s,
        smooth_root=not args.no_smooth_root,
        root_min_cutoff=args.root_min_cutoff, root_beta=args.root_beta,
        still_deadzone_mm_s=args.still_deadzone_mm_s or None)
    summary.to_csv(args.out_summary, index=False)
    print(f"[movement_metrics] summary -> {args.out_summary}")
    if args.out_per_frame:
        per_frame.to_csv(args.out_per_frame, index=False)
        print(f"[movement_metrics] per-frame -> {args.out_per_frame}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
