"""
pose/posture_3d.py
====================
Standing / seated / on the floor, per person and frame, from how high the
SHOULDERS are above the floor -- measured in 3D (ViTPose keypoints with
depth + this camera's own floor plane), so it does not depend on how far
the person is from the camera or on the viewing angle.

Why shoulders: with masked ViTPose, hips and legs hidden by the table have
no depth reading, the shoulders almost always do. Shoulder height above the
floor separates the three postures cleanly once body size is accounted for,
which is why the thresholds depend on the role (`pose.classify_age_group`):

                    adult          child
  standing     >= 1.15 m       >= 0.95 m
  seated       0.70-1.15 m     0.45-0.95 m
  on the floor  < 0.70 m        < 0.45 m     (sitting/kneeling on the floor, crouching)

(average adult shoulder height standing ~1.35-1.45 m, seated on a chair
~1.0 m; a 6-10 year old child ~0.9-1.1 m standing, ~0.65-0.75 m seated on a
child chair, ~0.9 m on an adult chair). These fixed thresholds are the
fallback: `classify_adaptive` first looks for the person's own seated and
standing shoulder levels and splits between them.

Stability: the shoulder height is median-filtered over `--window-s`
(default 1 s) before classifying, and a posture must last `--min-s`
(default 1.5 s) or it is absorbed by its neighbours -- no single-frame
flicker between labels.

Outputs:
  posture_<camera>.csv        per person per frame: shoulder height, posture
  posture_<camera>_elan.tsv   one row per posture episode, ready for ELAN's
                              "Import CSV / Tab-delimited Text File":
                              tier, begin (s), end (s), value
                              (tier = posture_id<N>_<role>). Times are video
                              time from the start of the recording: if the
                              ELAN file uses an audio offset, apply it when
                              importing or comparing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS_M = {  # (standing_min, seated_min)
    "adult": (1.15, 0.70),
    "child": (0.95, 0.45),
    "uncertain": (1.00, 0.60),
}
LABEL_IT = {"standing": "standing", "seated": "seated", "floor": "on the floor"}


def shoulder_height(df3d: pd.DataFrame, plane: dict) -> pd.DataFrame:
    """Height (m) of the mid-shoulder point above the floor plane, per
    person and frame; a single shoulder is used when only one has depth."""
    s = df3d[df3d["keypoint_name"].isin(["left_shoulder", "right_shoulder"])].dropna(subset=["X_mm", "Y_mm", "Z_mm"])
    mid = s.groupby(["global_person_id", "frame"])[["X_mm", "Y_mm", "Z_mm"]].mean()
    n = np.asarray(plane["normal"], float)
    o = np.asarray(plane["point"], float)
    h = (mid.to_numpy() - o) @ n / 1000.0
    return pd.DataFrame({"shoulder_h_m": h}, index=mid.index).reset_index()


def _absorb_short(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Replace runs shorter than min_len with the previous run's label."""
    out = labels.copy()
    if len(out) == 0:
        return out
    starts = np.r_[0, np.nonzero(out[1:] != out[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(out)]
    for s, e in zip(starts, ends):
        if e - s < min_len and s > 0:
            out[s:e] = out[s - 1]
    return out


def classify_heights(h: pd.Series, role: str, *, fps: float, window_s: float = 1.0, min_s: float = 1.5,
                     thresholds: dict | None = None) -> tuple[pd.Series, np.ndarray]:
    """Posture labels from a frame-indexed shoulder-height series (m above the
    floor, NaN = unknown). Returns (median-filtered heights, labels with None
    where unknown)."""
    thr = thresholds or THRESHOLDS_M
    stand_min, seat_min = thr.get(role, thr["uncertain"])
    h = h.rolling(max(1, int(window_s * fps)), center=True, min_periods=1).median()
    lab = np.where(h >= stand_min, "standing", np.where(h >= seat_min, "seated", "floor")).astype(object)
    valid = ~h.isna().to_numpy()
    lab[~valid] = None
    if valid.any():  # absorb short runs among the frames that have a value
        v = pd.Series(lab).ffill().bfill().to_numpy()
        v = _absorb_short(v, int(min_s * fps))
        lab = np.where(valid, v, None)
    return h, lab


def classify_adaptive(h: pd.Series, role: str, *, fps: float, window_s: float = 1.0, min_s: float = 1.5,
                      min_gap_m: float = 0.10, thresholds: dict | None = None) -> tuple[pd.Series, np.ndarray, dict]:
    """Seated vs standing from the person's OWN two shoulder-height levels.

    A fixed threshold fails for children: seated on an adult chair a child's
    shoulders are almost as high as when standing (10_individual_12: 0.90 m
    seated vs 1.08 m standing). Instead the (1 s median-filtered) shoulder
    heights of the session are split into two levels (2-means); when the two
    levels are at least `min_gap_m` apart and the lower one is not already a
    standing height, the midpoint separates seated from standing. Otherwise
    (the person never changes posture, e.g. always standing) the fixed
    role-dependent thresholds decide. 'On the floor' always uses the fixed
    lower threshold. Returns (smoothed heights, labels, info)."""
    thr = thresholds or THRESHOLDS_M
    stand_min, seat_min = thr.get(role, thr["uncertain"])
    hs = h.rolling(max(1, int(window_s * fps)), center=True, min_periods=1).median()
    v = hs.dropna().to_numpy()
    info = {"mode": "fixed", "levels_m": None, "split_m": stand_min}
    split = stand_min
    if len(v) > fps * 10:
        c = np.percentile(v, [25, 90]).astype(float)
        for _ in range(50):
            lab = np.abs(v[:, None] - c[None]).argmin(1)
            new = np.array([v[lab == k].mean() if (lab == k).any() else c[k] for k in (0, 1)])
            if np.allclose(new, c):
                break
            c = new
        info["levels_m"] = [round(float(x), 3) for x in c]
        if c[1] - c[0] >= min_gap_m and c[0] < stand_min:
            split = float(c.mean())
            info.update(mode="adaptive", split_m=round(split, 3))
    lab = np.where(hs >= split, "standing", np.where(hs >= seat_min, "seated", "floor")).astype(object)
    valid = ~hs.isna().to_numpy()
    lab[~valid] = None
    if valid.any():
        vv = pd.Series(lab).ffill().bfill().to_numpy()
        vv = _absorb_short(vv, int(min_s * fps))
        lab = np.where(valid, vv, None)
    return hs, lab, info


RATIO_STANDING_MIN = 0.85   # hip height / leg length (standing ~1.0, seated ~0.5-0.75)
RATIO_SEATED_MIN = 0.32


def classify_hip_ratio(ratio: pd.Series, shoulder_h: pd.Series, role: str, *, fps: float, window_s: float = 1.0,
                       min_s: float = 1.5) -> tuple[pd.Series, np.ndarray]:
    """Posture from how high the HIPS are relative to the person's own leg
    length (thigh + shin): ~1 standing, ~0.5 seated on a chair, low on the
    floor. Independent of body size, so a child seated on an adult chair
    (shoulders as high as a small child standing) is not called standing.
    Frames without a usable hip ratio fall back to the shoulder-height rule."""
    r = ratio.rolling(max(1, int(window_s * fps)), center=True, min_periods=1).median()
    lab = np.where(r >= RATIO_STANDING_MIN, "standing", np.where(r >= RATIO_SEATED_MIN, "seated", "floor")).astype(object)
    lab[r.isna().to_numpy()] = None
    _, lab_sh = classify_heights(shoulder_h, role, fps=fps, window_s=window_s, min_s=min_s)
    lab = np.where(pd.isna(lab), lab_sh, lab)
    valid = ~pd.isna(lab)
    if valid.any():
        v = pd.Series(lab).ffill().bfill().to_numpy()
        v = _absorb_short(v, int(min_s * fps))
        lab = np.where(valid, v, None)
    return r, lab


def classify_posture(df3d: pd.DataFrame, plane: dict, roles: dict[int, str], *, fps: float,
                     window_s: float = 1.0, min_s: float = 1.5, thresholds: dict | None = None) -> pd.DataFrame:
    thr = thresholds or THRESHOLDS_M
    sh = shoulder_height(df3d, plane)
    rows = []
    for pid, g in sh.groupby("global_person_id"):
        role = roles.get(int(pid), "uncertain")
        g = g.set_index("frame").sort_index()
        full = g.reindex(range(int(g.index.min()), int(g.index.max()) + 1))
        h, lab = classify_heights(full["shoulder_h_m"], role, fps=fps, window_s=window_s, min_s=min_s, thresholds=thr)
        rows.append(pd.DataFrame({"global_person_id": int(pid), "frame": full.index, "role": role,
                                  "shoulder_h_m": h.to_numpy(), "posture": lab}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["global_person_id", "frame", "role", "shoulder_h_m", "posture"])
    return out[out["posture"].notna()]


def elan_episodes(post: pd.DataFrame, fps: float, max_gap_s: float = 1.0) -> pd.DataFrame:
    """Contiguous runs of the same posture -> ELAN tab-delimited rows."""
    rows = []
    for (pid, role), g in post.groupby(["global_person_id", "role"]):
        g = g.sort_values("frame")
        f = g["frame"].to_numpy()
        lab = g["posture"].to_numpy()
        brk = np.r_[0, np.nonzero((lab[1:] != lab[:-1]) | (np.diff(f) > max_gap_s * fps))[0] + 1, len(f)]
        for s, e in zip(brk[:-1], brk[1:]):
            rows.append({"tier": f"posture_id{pid}_{role}", "begin_s": round(f[s] / fps, 3),
                         "end_s": round((f[e - 1] + 1) / fps, 3), "value": LABEL_IT.get(lab[s], lab[s])})
    return pd.DataFrame(rows, columns=["tier", "begin_s", "end_s", "value"])
