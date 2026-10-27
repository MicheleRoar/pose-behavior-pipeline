"""
pose/session_metrics.py
=========================
Per-person behaviour measures of a session, from BOTH cameras fused in one
3D frame (`pose.export_twin.load_session_people`: ViTPose skeletons,
bone-length corrected depth, camera B aligned on camera A):

  - speed (m/s) and total distance WALKED (m, `locomotion_path`: floor plane,
    1 s average, counted only beyond 15 cm moves -- seated fidgeting and
    residual depth noise do not accumulate); `distance_raw_m` keeps the
    per-frame sum for comparison
  - posture: standing / seated / on the floor, from the person's own two
    shoulder-height levels (`pose.posture_3d.classify_adaptive`); the hip
    height / leg length ratio is kept as a diagnostic column only: hips
    hidden behind the table come out too high to be trusted
  - distance from the child to each adult (m, on the floor), time spent close

Position used for speed and distance: the fused torso centre (camera B shifted
by the person's median A-B offset, then averaged with A) put through the
validated `pose.movement_metrics` chain -- outlier rejection, One Euro
smoothing, stillness deadzone. No "hold" here (that is only for drawing).

Outputs (in --out-dir, default <session>/metrics):
  metrics_per_frame.csv   pid, label, role, frame, t_s, x_m, z_m, shoulder_h_m,
                          posture, speed_ms, distance_m (cumulative),
                          dist_to_child_m (adults only)
  summary.csv             one row per person: distance, mean speed, % standing /
                          seated / on the floor, coverage; for adults: mean /
                          min distance to the child and % of time within
                          --close-m of the child
  posture_elan.tsv        posture episodes for ELAN (tier, begin, end, value)

Usage:
    cd src
    python -m pose.session_metrics --session-dir ".../Sessions/10_individual_12" \\
        --alignment-json .../calib/camera_b_to_camera_a_people.json --extra-pairs 3:3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pose.dwell_heatmap_3d import movement_from_roots
from pose.export_twin import fused_torso_mm, load_session_people, shoulder_mid_mm
import warnings

from pose.keypoints import COCO17
from pose.posture_3d import classify_adaptive

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN frames (person absent) are expected


def locomotion_path(x: np.ndarray, z: np.ndarray, fps: float, *, window_s: float = 1.0,
                    hold_m: float = 0.15) -> np.ndarray:
    """Cumulative distance WALKED (m) per frame, on the floor plane only:
    the position is averaged over `window_s` and the distance grows only when
    the person has moved more than `hold_m` from where they were last counted.
    Leaning, turning on the chair and the residual depth noise of a seated
    person stay below that, so they do not add up over a session (with the
    per-frame chain a child seated for 11 minutes still 'walked' ~6 m)."""
    s = pd.DataFrame({"x": x, "z": z}).rolling(max(1, int(window_s * fps)), center=True,
                                               min_periods=max(1, int(window_s * fps) // 3)).mean()
    out = np.zeros(len(s))
    tot, held = 0.0, None
    for i, (a, b) in enumerate(s.to_numpy()):
        if np.isfinite(a):
            if held is None:
                held = (a, b)
            else:
                d = np.hypot(a - held[0], b - held[1])
                if d > hold_m:
                    tot += d
                    held = (a, b)
        out[i] = tot
    return out


def hip_ratio(views: list[np.ndarray], fused: np.ndarray, to_w) -> pd.Series:
    """Hip height above the floor / the person's leg length (thigh + shin,
    medians over the session of the fused, bone-length-corrected skeleton)."""
    J = {k: COCO17.index(k) for k in COCO17}
    L = []
    for side in ("left", "right"):
        th = np.linalg.norm(fused[:, J[f"{side}_hip"]] - fused[:, J[f"{side}_knee"]], axis=1)
        sh = np.linalg.norm(fused[:, J[f"{side}_knee"]] - fused[:, J[f"{side}_ankle"]], axis=1)
        L.append(np.nanmedian(th) + np.nanmedian(sh))
    leg_m = np.nanmean(L) / 1000.0
    hips = np.nanmean(np.stack([np.nanmean(T[:, [J["left_hip"], J["right_hip"]]], axis=1) for T in views]), axis=0)
    hip_h = to_w(hips)[:, 1]
    return pd.Series(hip_h / leg_m if np.isfinite(leg_m) and leg_m > 0.3 else np.full(len(hip_h), np.nan)), leg_m


def main() -> None:
    p = argparse.ArgumentParser(description="Speed, distance, posture, child-adult distance (A+B fused).")
    p.add_argument("--session-dir", required=True)
    p.add_argument("--alignment-json", default=None)
    p.add_argument("--extra-pairs", default=None, help="A:B pairs to merge on top of the alignment, e.g. 3:3")
    p.add_argument("--pose-suffix", default="_vitpose")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--close-m", type=float, default=1.0, help="'close to the child' threshold")
    p.add_argument("--out-dir", default=None, help="default: <session>/metrics")
    a = p.parse_args()

    S = Path(a.session_dir)
    al = json.loads(Path(a.alignment_json).read_text()) if a.alignment_json else None
    ses = load_session_people(S, alignment=al, fps=a.fps, smooth_frames=1, pose_suffix=a.pose_suffix,
                              extra_pairs=a.extra_pairs, log="metrics")
    to_w, frames = ses["to_w"], ses["frames"]
    out_dir = Path(a.out_dir) if a.out_dir else S / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- fused torso -> world -> validated movement chain
    rows, heights = [], {}
    for pid, views in ses["views"].items():
        tor_w = to_w(fused_torso_mm(views)) * 1000.0          # world, mm (floor y=0, y up)
        ok = np.isfinite(tor_w).all(1)
        rows.append(pd.DataFrame({"global_person_id": pid, "frame": frames[ok],
                                  "X": tor_w[ok, 0], "Y": tor_w[ok, 1], "Z": tor_w[ok, 2]}))
        heights[pid] = pd.Series(to_w(shoulder_mid_mm(views))[:, 1], index=frames)
    roots = pd.concat(rows, ignore_index=True)
    mv = movement_from_roots(roots, a.fps).sort_values(["global_person_id", "frame"])
    mv["disp"] = mv["displacement_mm"].fillna(0.0)
    mv["distance_m"] = mv.groupby("global_person_id")["disp"].cumsum() / 1000.0
    mv["speed_ms"] = mv["velocity_mm_s"] / 1000.0

    # --- per person frame table
    tabs = []
    for pid in ses["views"]:
        role = ses["roles"].get(pid, "uncertain")
        ratio, leg_m = hip_ratio(ses["views"][pid], ses["sk"][pid], to_w)
        ratio.index = frames
        _, lab, pinfo = classify_adaptive(heights[pid], role, fps=a.fps)
        print(f"[metrics] {ses['labels'][pid]} ({role}): posture {pinfo['mode']}, shoulder levels "
              f"{pinfo['levels_m']} m, split at {pinfo['split_m']} m")
        t = pd.DataFrame({"pid": pid, "label": ses["labels"][pid], "role": role, "frame": frames,
                          "t_s": frames / a.fps, "shoulder_h_m": heights[pid].to_numpy(),
                          "hip_ratio": ratio.to_numpy(), "leg_m": round(float(leg_m), 3), "posture": lab})
        m = mv[mv["global_person_id"] == pid].set_index("frame")
        t = t.join(m[["X_mm", "Z_mm", "speed_ms", "distance_m"]], on="frame")
        t["x_m"], t["z_m"] = t.pop("X_mm") / 1000.0, t.pop("Z_mm") / 1000.0
        t["distance_raw_m"] = t["distance_m"].ffill().fillna(0.0)
        t["distance_m"] = locomotion_path(t["x_m"].to_numpy(), t["z_m"].to_numpy(), a.fps)
        tabs.append(t)
    per = pd.concat(tabs, ignore_index=True)

    # --- child <-> adults distance on the floor
    kids = [pid for pid in ses["views"] if ses["roles"].get(pid) == "child"]
    per["dist_to_child_m"] = np.nan
    if len(kids) == 1:
        kid = per[per["pid"] == kids[0]].set_index("frame")[["x_m", "z_m"]]
        for pid in ses["views"]:
            if pid == kids[0]:
                continue
            sel = per["pid"] == pid
            me = per.loc[sel].set_index("frame")[["x_m", "z_m"]]
            d = np.hypot(me["x_m"] - kid["x_m"].reindex(me.index), me["z_m"] - kid["z_m"].reindex(me.index))
            per.loc[sel, "dist_to_child_m"] = d.to_numpy()
    elif len(kids) > 1:
        print("[metrics] more than one child: child-adult distances not computed (group session)")
    per.to_csv(out_dir / "metrics_per_frame.csv", index=False)

    # --- summary
    summ = []
    for pid, g in per.groupby("pid"):
        seen = g["x_m"].notna()
        pst = g["posture"].value_counts(normalize=True) * 100
        row = {"pid": pid, "label": g["label"].iloc[0], "role": g["role"].iloc[0], "leg_m": g["leg_m"].iloc[0],
               "distance_m": round(float(g["distance_m"].max()), 1),
               "distance_raw_m": round(float(g["distance_raw_m"].max()), 1),
               "mean_speed_ms": round(float(g["speed_ms"].mean()), 3),
               "coverage_pct": round(100 * float(seen.mean()), 1)}
        for k in ("standing", "seated", "floor"):
            row[f"{k}_pct"] = round(float(pst.get(k, 0.0)), 1)
        d = g["dist_to_child_m"].dropna()
        if len(d):
            row.update(mean_dist_to_child_m=round(float(d.mean()), 2), min_dist_to_child_m=round(float(d.min()), 2),
                       **{f"within_{a.close_m:g}m_of_child_pct": round(100 * float((d <= a.close_m).mean()), 1)})
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(out_dir / "summary.csv", index=False)

    # --- ELAN
    from pose.posture_3d import elan_episodes
    el = elan_episodes(per.rename(columns={"pid": "global_person_id"}).dropna(subset=["posture"]), a.fps)
    el.to_csv(out_dir / "posture_elan.tsv", sep="\t", index=False)

    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(f"[metrics] wrote {out_dir}/metrics_per_frame.csv, summary.csv, posture_elan.tsv")


if __name__ == "__main__":
    main()
