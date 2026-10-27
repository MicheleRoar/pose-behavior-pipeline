"""
pose/classify_age_group.py
============================
Adult vs child per tracked person, from the person's real body size in 3D.

Replaces `pose/classify_person_roles.py` for the ViTPose pipeline. That one
worked on MediaPipe-lite keypoints whose depth was also read on hidden parts
(hips behind the table got the table's depth), which made bone lengths
unreliable (on 10_individual_12 one variant called a therapist the child and
gave the father a 2.5 m "height").

How
----
For each person, the 3D length of a few rigid segments between COCO-17
joints is measured in every frame where BOTH ends have a real depth reading
(`keypoints_3d.csv` from `pose.depth`; with ViTPose keypoints outside the
person's own mask have no depth, so occluded joints never enter). Per
segment, the median over the session is divided by a typical ADULT length
for the same joint pair; the person's size index is the median of those
ratios over the segments with enough samples. Bone lengths do not change with
posture, so this works on seated people too.

  index >= --adult-min (0.85)  -> adult
  index <= --child-max (0.78)  -> child
  in between                   -> uncertain (shown grey, decide by eye)

The adult reference lengths are joint-centre distances for an average adult
(shoulder width between the two shoulder keypoints ~ 330 mm, upper arm
~ 290 mm, forearm ~ 250 mm, thigh ~ 410 mm, shin ~ 390 mm, shoulder-to-hip
on one side ~ 470 mm); a 6-10 year old is roughly 0.6-0.75 of these. They
are a scale, not a measurement: the thresholds are what matters, and the
printed per-segment ratios show how clear-cut each decision is.

Output: `age_group.csv` (one row per person: size_index, role, segments
used, per-segment ratios and sample counts) and the `--roles` string the
other tools accept, e.g. `0:adult,1:adult,2:child,3:adult`.

Usage:
    cd src
    python -m pose.classify_age_group --keypoints-3d .../pose/camera_a_vitpose/keypoints_3d.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ADULT_REF_MM = {
    ("left_shoulder", "right_shoulder"): 330.0,
    ("left_shoulder", "left_elbow"): 290.0,
    ("right_shoulder", "right_elbow"): 290.0,
    ("left_elbow", "left_wrist"): 250.0,
    ("right_elbow", "right_wrist"): 250.0,
    ("left_hip", "left_knee"): 410.0,
    ("right_hip", "right_knee"): 410.0,
    ("left_knee", "left_ankle"): 390.0,
    ("right_knee", "right_ankle"): 390.0,
    ("left_shoulder", "left_hip"): 470.0,
    ("right_shoulder", "right_hip"): 470.0,
}


def segment_lengths(df3d: pd.DataFrame, *, max_dz_mm: float = 350.0) -> pd.DataFrame:
    """Long table: person, frame, segment, length_mm -- only where both ends
    have depth and their depths are compatible (a joint whose depth jumps to
    the wall behind would otherwise make a 1 m 'forearm')."""
    p = df3d.dropna(subset=["X_mm", "Y_mm", "Z_mm"]).pivot_table(
        index=["global_person_id", "frame"], columns="keypoint_name", values=["X_mm", "Y_mm", "Z_mm"])
    out = []
    for (a, b), ref in ADULT_REF_MM.items():
        if ("X_mm", a) not in p.columns or ("X_mm", b) not in p.columns:
            continue
        A = np.stack([p[(ax, a)] for ax in ("X_mm", "Y_mm", "Z_mm")], 1)
        B = np.stack([p[(ax, b)] for ax in ("X_mm", "Y_mm", "Z_mm")], 1)
        L = np.linalg.norm(A - B, axis=1)
        ok = np.isfinite(L) & (np.abs(A[:, 2] - B[:, 2]) <= max_dz_mm) & (L > 0.25 * ref) & (L < 2.0 * ref)
        seg = pd.DataFrame({"global_person_id": p.index.get_level_values(0)[ok],
                            "frame": p.index.get_level_values(1)[ok],
                            "segment": f"{a}-{b}", "length_mm": L[ok], "ref_mm": ref})
        out.append(seg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["global_person_id", "frame", "segment", "length_mm", "ref_mm"])


def classify(df3d: pd.DataFrame, *, min_samples: int = 60, min_segments: int = 3,
             adult_min: float = 0.85, child_max: float = 0.78) -> pd.DataFrame:
    seg = segment_lengths(df3d)
    rows = []
    for pid, g in seg.groupby("global_person_id"):
        per = g.groupby("segment").agg(n=("length_mm", "size"), med=("length_mm", "median"), ref=("ref_mm", "first"))
        per = per[per["n"] >= min_samples]
        ratios = (per["med"] / per["ref"]).sort_values()
        row = {"global_person_id": int(pid), "n_segments": int(len(ratios)),
               "size_index": float(ratios.median()) if len(ratios) else np.nan}
        if len(ratios) < min_segments or not np.isfinite(row["size_index"]):
            row["role"] = "uncertain"
        elif row["size_index"] >= adult_min:
            row["role"] = "adult"
        elif row["size_index"] <= child_max:
            row["role"] = "child"
        else:
            row["role"] = "uncertain"
        for s, r in ratios.items():
            row[f"ratio:{s}"] = round(float(r), 3)
            row[f"n:{s}"] = int(per.loc[s, "n"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("global_person_id")


def apply_expected_children(table: pd.DataFrame, n_children: int, min_gap: float = 0.10) -> pd.DataFrame:
    """Session-relative fallback for when the absolute scale is off (e.g.
    noisy keypoints shrinking every segment): in an individual session the
    child is the patient, so `n_children=1`. The n smallest size indices
    become `child`, the rest `adult` -- only if the n-th smallest is at
    least `min_gap` (relative) below the next one; otherwise roles are left
    as they were and a warning is printed. The absolute verdict is kept in
    `role_absolute` for comparison."""
    t = table.copy()
    t["role_absolute"] = t["role"]
    v = t.dropna(subset=["size_index"]).sort_values("size_index")
    if len(v) <= n_children:
        return t
    last_child, first_adult = v["size_index"].iloc[n_children - 1], v["size_index"].iloc[n_children]
    if first_adult - last_child < min_gap * first_adult:
        print(f"[age_group] WARNING: no clear gap between child {last_child:.3f} and adult {first_adult:.3f} "
              f"-- relative rule not applied")
        return t
    kids = set(v["global_person_id"].iloc[:n_children])
    t["role"] = ["child" if pid in kids else "adult" for pid in t["global_person_id"]]
    return t


def roles_string(table: pd.DataFrame) -> str:
    return ",".join(f"{int(r.global_person_id)}:{r.role}" for r in table.itertuples())


def main() -> None:
    p = argparse.ArgumentParser(description="Adult vs child from 3D bone lengths -- see module docstring.")
    p.add_argument("--keypoints-3d", required=True, help="keypoints_3d.csv (ViTPose pipeline)")
    p.add_argument("--out", default=None, help="default: age_group.csv next to the input")
    p.add_argument("--adult-min", type=float, default=0.85)
    p.add_argument("--child-max", type=float, default=0.78)
    p.add_argument("--min-samples", type=int, default=60, help="frames needed per segment")
    p.add_argument("--expected-children", type=int, default=None,
                   help="session-relative rule: the N smallest are children if clearly separated (1 = individual session)")
    a = p.parse_args()
    df = pd.read_csv(a.keypoints_3d)
    t = classify(df, min_samples=a.min_samples, adult_min=a.adult_min, child_max=a.child_max)
    if a.expected_children:
        t = apply_expected_children(t, a.expected_children)
    out = Path(a.out) if a.out else Path(a.keypoints_3d).with_name("age_group.csv")
    t.to_csv(out, index=False)
    cols = ["global_person_id", "size_index", "role", "n_segments"] + (["role_absolute"] if "role_absolute" in t else [])
    print(t[cols].round(3).to_string(index=False))
    print(f"\n[age_group] roles: {roles_string(t)}\n[age_group] wrote {out}")


if __name__ == "__main__":
    main()
