"""
pose/align_cameras_from_people.py
===================================
Estimates the rigid transform camera_b -> camera_a AUTOMATICALLY, using
the people in the session as the calibration target instead of points
clicked by hand on the table (`pose/table_calibration.py`).

Why this works
---------------
Both Azure Kinects give metric depth, so every tracked person already has
a 3D trajectory in each camera's own frame (`keypoints_3d.csv` from
`pose.run_depth`). The same person at the same instant is at the same
physical spot, so thousands of (camera_b position, camera_a position)
pairs, spread over the whole room and the whole session, constrain the
transform far better than 4-6 clicks on a small, flat object (the
sheet-of-paper calibration on 10_individual_12 had a 73.7 mm RMSE on
points spanning ~30 cm, which leaves the TILT poorly determined).

What it does
-------------
1. Torso root per (person, frame): median of shoulders + hips whose depth
   agrees with the torso median (+-`z_tol_mm`) -- more robust than the
   hip midpoint, which on a seated person is often hidden by the table.
2. Which camera_a id is which camera_b id is NOT assumed: every one-to-one
   assignment between the ids with enough frames is tried (people per
   session are few), a robust rigid fit is done for each, and the
   assignment whose matched people coincide best wins. An initial
   transform (`--init-transform`, e.g. the table_calibration one) is
   optional; if given it is also used as a starting hypothesis.
3. Frame offset: the fit is repeated for camera_b shifted by
   -`max_offset_frames`..+`max_offset_frames` frames and the offset with
   the lowest residual is kept. The whole residual-vs-offset curve is
   written to the report: a flat curve means the people did not move
   enough to tell (offset not determined), not that the offset is 0.
4. Floor (optional, `--cam-a-pointcloud`/`--cam-b-pointcloud` from
   `table_calibration extract`): each camera's floor plane is fitted
   automatically (`room_coverage.fit_floor_plane`) and the two normals
   are added to the rotation fit as one extra, heavily weighted direction
   pair -- it pins the tilt, which people sitting in one spot cannot.
   The angle between the two floors after alignment is reported either way.

Known limits
-------------
- Front/back offset: camera_a sees one side of a body, camera_b the other,
  so the same person's torso is measured ~10-30 cm apart along the line
  between the cameras. The trimmed fit absorbs part of it; the residual
  never goes to zero and should not be expected to.
- A session where everyone stays seated in one place gives poorly spread
  correspondences: check `n_pairs`, the spread of the matched points
  (`spread_mm`) and the floor angle before trusting the result.
- Garbage in, garbage out: wrong per-camera ids (stale pose on a re-merged
  MaskDir) make whole people jump, and the fit inherits it.

Output JSON has the same `rotation_3x3`/`translation_mm` keys as
`table_calibration register`, so `table_calibration apply` and
`room_coverage` use it unchanged.

Usage:
    cd src
    python -m pose.align_cameras_from_people \\
        --camera-a-csv .../pose/camera_a/keypoints_3d.csv \\
        --camera-b-csv .../pose/camera_b/keypoints_3d.csv \\
        --init-transform .../camera_b_to_camera_a_transform.json \\
        --cam-a-pointcloud .../calib/cam_a_pointcloud.npy \\
        --cam-b-pointcloud .../calib/cam_b_pointcloud.npy \\
        --out-json .../calib/camera_b_to_camera_a_people.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

TORSO_KEYPOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------

def torso_roots(df: pd.DataFrame, *, z_tol_mm: float = 250.0, min_points: int = 2) -> pd.DataFrame:
    """One robust 3D torso position per (global_person_id, frame).
    Returns columns global_person_id, frame, X, Y, Z (mm)."""
    t = df[df["keypoint_name"].isin(TORSO_KEYPOINTS)].dropna(subset=["X_mm", "Y_mm", "Z_mm"])
    zmed = t.groupby(["global_person_id", "frame"])["Z_mm"].transform("median")
    t = t[(t["Z_mm"] - zmed).abs() <= z_tol_mm]
    g = t.groupby(["global_person_id", "frame"])
    out = g[["X_mm", "Y_mm", "Z_mm"]].median()
    out["n"] = g.size()
    out = out[out["n"] >= min_points].drop(columns="n").reset_index()
    return out.rename(columns={"X_mm": "X", "Y_mm": "Y", "Z_mm": "Z"})


def _tracks(roots: pd.DataFrame, min_frames: int) -> dict[int, pd.DataFrame]:
    out = {}
    for pid, g in roots.groupby("global_person_id"):
        if len(g) >= min_frames:
            out[int(pid)] = g.set_index("frame")[["X", "Y", "Z"]]
    return out


def load_transform(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(Path(path).read_text())
    return np.asarray(d["rotation_3x3"], float), np.asarray(d["translation_mm"], float)


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def weighted_kabsch(src: np.ndarray, dst: np.ndarray, *, dir_src: np.ndarray | None = None,
                    dir_dst: np.ndarray | None = None, dir_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Rigid R, t minimizing sum ||R src + t - dst||^2, optionally plus a
    direction pair (unit vectors, e.g. the two floor normals) weighted as
    `dir_weight` extra point pairs at unit distance."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    if dir_src is not None and dir_weight > 0:
        H = H + dir_weight * np.outer(dir_src, dir_dst)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _pairs(tr_a, tr_b, assignment, offset, stride):
    """Stacked (src=camera_b, dst=camera_a) correspondences for an assignment."""
    src, dst = [], []
    for ia, ib in assignment:
        b = tr_b[ib]
        b = b.set_axis(b.index + offset)
        j = tr_a[ia].join(b, rsuffix="_b", how="inner")
        if stride > 1:
            j = j.iloc[::stride]
        dst.append(j[["X", "Y", "Z"]].to_numpy())
        src.append(j[["X_b", "Y_b", "Z_b"]].to_numpy())
    if not src:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.vstack(src), np.vstack(dst)


def robust_fit(src, dst, *, keep: float = 0.6, iters: int = 15, floor=None, floor_weight_frac: float = 0.0):
    """Trimmed rigid fit: refit on the best `keep` fraction of pairs until
    stable. Returns R, t, residuals (all pairs), inlier mask."""
    kw = {}
    if floor is not None and floor_weight_frac > 0:
        # weight relative to the point spread, so the floor counts like a
        # fixed fraction of the point pairs regardless of how many there are
        spread = np.sum((src - src.mean(0)) ** 2)
        kw = dict(dir_src=floor[1], dir_dst=floor[0], dir_weight=floor_weight_frac * spread)
    R, t = weighted_kabsch(src, dst, **kw)
    sel = np.ones(len(src), bool)
    for _ in range(iters):
        r = np.linalg.norm(src @ R.T + t - dst, axis=1)
        new = r <= np.quantile(r, keep)
        if np.array_equal(new, sel):
            break
        sel = new
        if floor is not None and floor_weight_frac > 0:
            spread = np.sum((src[sel] - src[sel].mean(0)) ** 2)
            kw["dir_weight"] = floor_weight_frac * spread
        R, t = weighted_kabsch(src[sel], dst[sel], **kw)
    r = np.linalg.norm(src @ R.T + t - dst, axis=1)
    return R, t, r, sel


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def robust_fit_floor_locked(src, dst, planes, *, keep: float = 0.6, iters: int = 15):
    """Rigid fit with the two FLOORS forced to coincide: B's floor normal is
    rotated onto A's and B's floor plane put on A's (that fixes tilt, roll and
    height exactly, from tens of thousands of floor points); only the yaw
    about the vertical and the horizontal shift are estimated from the people,
    by a trimmed 2D fit on their floor-projected positions. Far better
    conditioned than a free 3D fit when people mostly sit in a few places:
    a few seated people constrain yaw/shift well, tilt hardly at all."""
    pa, pb = planes
    nA, oA = np.asarray(pa["normal"], float), np.asarray(pa["point"], float)
    nB, oB = np.asarray(pb["normal"], float), np.asarray(pb["point"], float)
    R1 = _rot_between(nB, nA)
    ex = np.array([1.0, 0.0, 0.0]) - nA * nA[0]
    ex /= np.linalg.norm(ex)
    ey = np.cross(nA, ex)
    E = np.stack([ex, ey, nA])                         # A-floor basis (rows)
    hb0 = float((R1 @ oB - oA) @ nA)                   # B floor height in A-floor coords
    a2 = (dst - oA) @ E.T                              # (u, v, h) of A points
    b2 = (src @ R1.T - oA) @ E.T                       # (u, v, h) of rotated B points

    def fit2(sel):
        A_, B_ = a2[sel, :2], b2[sel, :2]
        ca, cb = A_.mean(0), B_.mean(0)
        U, _, Vt = np.linalg.svd((B_ - cb).T @ (A_ - ca))
        R2 = Vt.T @ U.T
        if np.linalg.det(R2) < 0:
            Vt[-1] *= -1
            R2 = Vt.T @ U.T
        return R2, ca - R2 @ cb

    sel = np.ones(len(src), bool)
    for _ in range(iters):
        R2, t2 = fit2(sel)
        M = np.eye(3)
        M[:2, :2] = R2
        R = E.T @ M @ E @ R1
        t = oA + E.T @ np.r_[t2, -hb0] - E.T @ M @ E @ oA
        r = np.linalg.norm(src @ R.T + t - dst, axis=1)
        new = r <= np.quantile(r, keep)
        if np.array_equal(new, sel):
            break
        sel = new
    r = np.linalg.norm(src @ R.T + t - dst, axis=1)
    return R, t, r, sel


def _score(r: np.ndarray, keep: float) -> float:
    """Mean of the best `keep` fraction of residuals (pooled)."""
    k = max(1, int(len(r) * keep))
    return float(np.sort(r)[:k].mean())


def _assignment_score(tr_a, tr_b, assign, R, t, offset, stride) -> tuple[float, list[float]]:
    """Mean over the matched PAIRS of each pair's median residual. Scored
    per pair on purpose: a pooled trimmed score lets the fit quietly drop
    a whole wrongly matched person as 'outliers'; here every pair has to
    fit for the assignment to score well."""
    meds = []
    for ia, ib in assign:
        s, d = _pairs(tr_a, tr_b, [(ia, ib)], offset, stride)
        meds.append(float(np.median(np.linalg.norm(s @ R.T + t - d, axis=1))))
    return float(np.mean(meds)), meds


def candidate_assignments(ids_a, ids_b, min_people: int):
    """All one-to-one assignments between subsets of size >= min_people."""
    n = min(len(ids_a), len(ids_b))
    for size in range(n, min_people - 1, -1):
        for sub_a in itertools.combinations(ids_a, size):
            for perm_b in itertools.permutations(ids_b, size):
                yield tuple(zip(sub_a, perm_b))


def _cooccur(tr_a, tr_b, ia, ib) -> int:
    return len(tr_a[ia].index.intersection(tr_b[ib].index))


# ----------------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------------

def align_cameras(
    df_a: pd.DataFrame, df_b: pd.DataFrame, *,
    init: tuple[np.ndarray, np.ndarray] | None = None,
    floor_normals: tuple[np.ndarray, np.ndarray] | None = None,
    floor_weight_frac: float = 0.05,
    floor_planes: tuple[dict, dict] | None = None,
    min_frames: int = 300, min_people: int = 2, min_cooccur: int = 300,
    keep: float = 0.6, search_stride: int = 5, max_offset_frames: int = 30,
    offset_step: int = 1, size_tolerance: float = 1.25,
) -> dict:
    if floor_planes is not None:
        def FIT(src, dst):
            return robust_fit_floor_locked(src, dst, floor_planes, keep=keep)
    else:
        def FIT(src, dst):
            return robust_fit(src, dst, keep=keep, floor=floor_normals, floor_weight_frac=floor_weight_frac)

    tr_a = _tracks(torso_roots(df_a), min_frames)
    tr_b = _tracks(torso_roots(df_b), min_frames)
    ids_a, ids_b = sorted(tr_a), sorted(tr_b)
    if len(ids_a) < min_people or len(ids_b) < min_people:
        raise ValueError(f"Not enough tracked people (A: {ids_a}, B: {ids_b}, need >= {min_people} each "
                         f"with >= {min_frames} frames)")

    # 1) assignment search at offset 0, coarse stride for speed
    results = []
    for assign in candidate_assignments(ids_a, ids_b, min_people):
        if any(_cooccur(tr_a, tr_b, ia, ib) < min_cooccur for ia, ib in assign):
            continue
        src, dst = _pairs(tr_a, tr_b, assign, 0, search_stride)
        if len(src) < 50:
            continue
        R, t, r, _ = FIT(src, dst)
        sc, meds = _assignment_score(tr_a, tr_b, assign, R, t, 0, search_stride)
        results.append({"assignment": assign, "score_mm": sc, "pair_medians_mm": meds, "n_pairs": len(src)})
    if not results:
        raise ValueError("No assignment had enough co-occurring frames -- lower --min-cooccur?")

    # A smaller subset of people always fits at least as well, so: among
    # the assignments scoring within `size_tolerance` of the best one, take
    # the one matching the most people (then the lowest score).
    results.sort(key=lambda d: d["score_mm"])
    near = [d for d in results if d["score_mm"] <= size_tolerance * results[0]["score_mm"]]
    best = max(near, key=lambda d: (len(d["assignment"]), -d["score_mm"]))

    init_check = None
    if init is not None:
        R0, t0 = init
        src, dst = _pairs(tr_a, tr_b, best["assignment"], 0, search_stride)
        init_check = {"score_mm_with_initial_transform":
                      _assignment_score(tr_a, tr_b, best["assignment"], R0, t0, 0, search_stride)[0]}

    # 2) frame offset scan for the chosen assignment
    curve = []
    for off in range(-max_offset_frames, max_offset_frames + 1, offset_step):
        src, dst = _pairs(tr_a, tr_b, best["assignment"], off, search_stride)
        if len(src) < 50:
            continue
        Ro, to_, _, _ = FIT(src, dst)
        curve.append((off, _assignment_score(tr_a, tr_b, best["assignment"], Ro, to_, off, search_stride)[0]))
    best_off, best_off_score = min(curve, key=lambda c: c[1])
    zero_score = dict(curve).get(0, np.nan)

    # 3) final fit, all frames, at the chosen offset
    src, dst = _pairs(tr_a, tr_b, best["assignment"], best_off, 1)
    R, t, r, sel = FIT(src, dst)

    per_pair = []
    for ia, ib in best["assignment"]:
        s, d = _pairs(tr_a, tr_b, [(ia, ib)], best_off, 1)
        rr = np.linalg.norm(s @ R.T + t - d, axis=1)
        per_pair.append({"id_a": ia, "id_b": ib, "n_frames": int(len(rr)),
                         "median_mm": float(np.median(rr)), "p90_mm": float(np.percentile(rr, 90))})

    report = {
        "method": "people_as_calibration_target",
        "rotation_3x3": R.tolist(),
        "translation_mm": t.tolist(),
        "assignment": [{"id_a": a, "id_b": b} for a, b in best["assignment"]],
        "per_pair": per_pair,
        "n_pairs": int(len(src)),
        "inlier_fraction": float(sel.mean()),
        "median_residual_mm": float(np.median(r)),
        "trimmed_mean_residual_mm": _score(r, keep),
        "assignment_score_mm": _assignment_score(tr_a, tr_b, best["assignment"], R, t, best_off, 1)[0],
        "spread_mm": (dst[sel].std(0)).tolist(),
        "frame_offset_b_to_a": int(best_off),
        "offset_curve": [{"offset": o, "score_mm": s} for o, s in curve],
        "offset_gain_vs_zero_mm": float(zero_score - best_off_score) if np.isfinite(zero_score) else None,
        "assignment_candidates_top5": [
            {"assignment": [list(p) for p in d["assignment"]], "score_mm": d["score_mm"],
             "pair_medians_mm": d["pair_medians_mm"], "n_pairs": d["n_pairs"]}
            for d in results[:5]
        ],
        "ids_a": ids_a, "ids_b": ids_b,
    }
    if floor_normals is not None:
        na, nb = floor_normals
        report["floor_angle_deg_after"] = float(np.degrees(np.arccos(np.clip(na @ (R @ nb), -1, 1))))
        report["floor_mode"] = "locked" if floor_planes is not None else f"soft (weight {floor_weight_frac})"
    if init is not None:
        R0, t0 = init
        report["initial_transform"] = {
            **init_check,
            "rotation_diff_deg": rotation_angle_deg(R @ R0.T),
            "translation_diff_mm": float(np.linalg.norm(t - t0)),
        }
        if floor_normals is not None:
            na, nb = floor_normals
            report["initial_transform"]["floor_angle_deg"] = float(
                np.degrees(np.arccos(np.clip(na @ (R0 @ nb), -1, 1))))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="camera_b -> camera_a rigid transform from the people in the "
                                            "session (no clicks) -- see module docstring.")
    p.add_argument("--camera-a-csv", required=True, help="camera_a keypoints_3d.csv (its own frame)")
    p.add_argument("--camera-b-csv", required=True, help="camera_b keypoints_3d.csv (ITS OWN frame, not *_in_camera_a_frame)")
    p.add_argument("--out-json", required=True)
    p.add_argument("--init-transform", default=None, help="optional table_calibration transform, for comparison/seeding")
    p.add_argument("--cam-a-pointcloud", default=None, help="optional cam_a_pointcloud.npy (table_calibration extract)")
    p.add_argument("--cam-b-pointcloud", default=None, help="optional cam_b_pointcloud.npy")
    p.add_argument("--floor-mode", default="locked", choices=["locked", "soft"],
                   help="locked: floors forced to coincide, people fix only yaw + horizontal shift (default); "
                        "soft: free 3D fit with the floor as a weighted hint")
    p.add_argument("--floor-weight-frac", type=float, default=0.05,
                   help="soft mode only: weight of the floor-normal hint relative to the point spread")
    p.add_argument("--min-frames", type=int, default=300, help="ignore ids tracked in fewer frames")
    p.add_argument("--min-people", type=int, default=2)
    p.add_argument("--min-cooccur", type=int, default=300, help="min frames both ids of a pair are visible")
    p.add_argument("--keep", type=float, default=0.6, help="fraction of pairs kept by the trimmed fit")
    p.add_argument("--search-stride", type=int, default=5, help="frame subsampling during the search")
    p.add_argument("--max-offset-frames", type=int, default=30)
    p.add_argument("--size-tolerance", type=float, default=1.25,
                   help="an assignment with more people is preferred if its score is within this factor of the best")
    a = p.parse_args()

    df_a = pd.read_csv(a.camera_a_csv)
    df_b = pd.read_csv(a.camera_b_csv)
    init = load_transform(a.init_transform) if a.init_transform else None

    floor = None
    if a.cam_a_pointcloud and a.cam_b_pointcloud:
        from pose.room_coverage import _valid_points, fit_floor_plane
        pa = fit_floor_plane(_valid_points(np.load(a.cam_a_pointcloud)))
        pb = fit_floor_plane(_valid_points(np.load(a.cam_b_pointcloud)))
        print(f"[align] floor A: rmse {pa['rmse_mm']:.1f} mm, {pa['n_inliers']} inliers | "
              f"floor B: rmse {pb['rmse_mm']:.1f} mm, {pb['n_inliers']} inliers")
        floor = (pa["normal"], pb["normal"])
        planes = (pa, pb) if a.floor_mode == "locked" else None
    else:
        planes = None

    rep = align_cameras(df_a, df_b, init=init, floor_normals=floor, floor_weight_frac=a.floor_weight_frac,
                        floor_planes=planes,
                        min_frames=a.min_frames, min_people=a.min_people, min_cooccur=a.min_cooccur,
                        keep=a.keep, search_stride=a.search_stride, max_offset_frames=a.max_offset_frames,
                        size_tolerance=a.size_tolerance)
    rep["camera_a_csv"], rep["camera_b_csv"] = a.camera_a_csv, a.camera_b_csv
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(rep, indent=2))

    print(f"[align] assignment (A<->B): " + ", ".join(f"{d['id_a']}<->{d['id_b']}" for d in rep["assignment"]))
    for d in rep["per_pair"]:
        print(f"         A{d['id_a']}<->B{d['id_b']}: median {d['median_mm']:.0f} mm, p90 {d['p90_mm']:.0f} mm "
              f"({d['n_frames']} frames)")
    print(f"[align] per-pair score {rep['assignment_score_mm']:.0f} mm, trimmed mean residual {rep['trimmed_mean_residual_mm']:.0f} mm, "
          f"frame offset {rep['frame_offset_b_to_a']} (gain vs 0: {rep['offset_gain_vs_zero_mm']:.1f} mm)")
    print("[align] best assignments (score = mean of per-pair medians):")
    for c in rep["assignment_candidates_top5"]:
        pairs = ", ".join(f"A{x}<->B{y}" for x, y in c["assignment"])
        meds = ", ".join(f"{m:.0f}" for m in c["pair_medians_mm"])
        print(f"         {c['score_mm']:6.0f} mm  [{pairs}]  per pair: {meds}")
    if "floor_angle_deg_after" in rep:
        print(f"[align] floor angle after alignment: {rep['floor_angle_deg_after']:.1f} deg")
    if "initial_transform" in rep:
        it = rep["initial_transform"]
        print(f"[align] vs initial transform: rotation differs {it['rotation_diff_deg']:.1f} deg, "
              f"translation {it['translation_diff_mm']:.0f} mm; initial score "
              f"{it['score_mm_with_initial_transform']:.0f} mm"
              + (f", initial floor angle {it['floor_angle_deg']:.1f} deg" if "floor_angle_deg" in it else ""))
    print(f"[align] wrote {a.out_json}")


if __name__ == "__main__":
    main()
