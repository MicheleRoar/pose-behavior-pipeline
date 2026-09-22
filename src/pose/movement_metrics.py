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
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

_ROOT_KEYPOINTS = ("left_hip", "right_hip")


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


def compute_movement_metrics(df_3d: pd.DataFrame, *, fps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns `(per_frame, summary)`:

    `per_frame`: one row per (session, person, frame) with root X/Y/Z
    (mm), `displacement_mm` (3D distance from the PREVIOUS frame with a
    valid root for this person -- NaN for the first valid frame or any
    frame with no valid root itself), `dt_s`, and `velocity_mm_s`.

    `summary`: one row per (session, person) with `total_distance_mm`,
    `mean_velocity_mm_s`, `velocity_std_mm_s` (the "movement
    variability" figure), `frames_with_root`, `total_frames`, and
    `coverage_fraction` -- always report `coverage_fraction` alongside
    the distance/velocity numbers: a low one means those numbers are
    based on a small, possibly unrepresentative slice of the session
    (see module docstring)."""
    root = _root_position(df_3d)
    span_by_person = df_3d.groupby(["session_id", "global_person_id"])["frame"].agg(
        lambda s: int(s.max() - s.min() + 1))

    out_frames = []
    summaries = []
    for (session_id, pid), g in root.groupby(["session_id", "global_person_id"], sort=False):
        g = g.sort_values("frame").reset_index(drop=True)
        pos = g[["X_mm", "Y_mm", "Z_mm"]].to_numpy()
        valid = ~np.isnan(pos).any(axis=1)
        frame = g["frame"].to_numpy()

        displacement = np.full(len(g), np.nan)
        dt_s = np.full(len(g), np.nan)
        velocity = np.full(len(g), np.nan)

        last_valid_idx = None
        for i in range(len(g)):
            if not valid[i]:
                continue
            if last_valid_idx is not None:
                d = float(np.linalg.norm(pos[i] - pos[last_valid_idx]))
                dt = (frame[i] - frame[last_valid_idx]) / fps
                displacement[i] = d
                dt_s[i] = dt
                velocity[i] = d / dt if dt > 0 else np.nan
            last_valid_idx = i

        g = g.copy()
        g["displacement_mm"] = displacement
        g["dt_s"] = dt_s
        g["velocity_mm_s"] = velocity
        out_frames.append(g)

        n_valid = int(valid.sum())
        total_frames = int(span_by_person.get((session_id, pid), n_valid))
        summaries.append({
            "session_id": session_id,
            "global_person_id": pid,
            "total_distance_mm": float(np.nansum(displacement)),
            "mean_velocity_mm_s": float(np.nanmean(velocity)) if n_valid > 1 else np.nan,
            "velocity_std_mm_s": float(np.nanstd(velocity)) if n_valid > 1 else np.nan,
            "frames_with_root": n_valid,
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
    args = parser.parse_args()

    df_3d = pd.read_csv(args.keypoints_3d_csv)
    per_frame, summary = compute_movement_metrics(df_3d, fps=args.fps)
    summary.to_csv(args.out_summary, index=False)
    print(f"[movement_metrics] summary -> {args.out_summary}")
    if args.out_per_frame:
        per_frame.to_csv(args.out_per_frame, index=False)
        print(f"[movement_metrics] per-frame -> {args.out_per_frame}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
