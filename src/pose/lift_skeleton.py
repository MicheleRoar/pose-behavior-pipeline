"""
pose/lift_skeleton.py
=======================
Fixes the depth of individual joints using the skeleton itself.

Why joints "fly away"
----------------------
`pose/depth.py` reads ONE depth pixel per keypoint. The 2D keypoint from
ViTPose is usually right, but on thin parts (wrists, elbows, ankles) and at
silhouette edges that single pixel often hits what is behind or in front of
the limb -- the wall, the table edge, another person. The 2D position stays
correct, only the DEPTH is wrong, so the 3D joint slides along the camera's
viewing ray: in the 3D view it shoots off towards (or away from) the camera,
drawing long lines across the room.

What this does
---------------
1. Camera model: the pinhole intrinsics (fx, fy, cx, cy) are fitted from the
   camera's own point cloud (every valid pixel gives u = fx*X/Z + cx), so any
   2D keypoint can be turned into its viewing ray, even where depth is missing.
2. Per person, the real length of each limb bone (upper arm, forearm, thigh,
   shin) is measured once, as the median over the frames where both ends
   have a consistent depth.
3. Torso joints (shoulders, hips) whose depth disagrees with the torso's
   median depth by more than `torso_tol_mm` are put back at the torso depth,
   along their own ray.
4. Limbs are rebuilt from the torso outwards (shoulder -> elbow -> wrist,
   hip -> knee -> ankle; head points from the shoulders): a child joint is
   placed on ITS viewing ray at exactly the bone length from its parent. Of
   the two possible points the one at the range closest to the parent's is
   kept (the measured depth, having failed the bone check, is not trusted). A measured joint is kept
   as is when it already gives a bone within +-`bone_tol` of the real length.

Joints hidden in 2D-only (no depth because outside the person's mask, e.g.
legs behind the table) also get a 3D position this way: the skeleton is
complete and limb lengths stay constant.

Input/output: the long `keypoints_3d.csv` format (x, y pixel + X_mm, Y_mm,
Z_mm), plus a `lifted` column (True where a joint's 3D was corrected).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

from pose.keypoints import COCO17

BONES = [  # parent -> child, processed in this order
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
TORSO = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
HEAD = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
DEFAULT_BONE_MM = {"upper_arm": 290, "forearm": 250, "thigh": 410, "shin": 390}


def fit_intrinsics(point_cloud: np.ndarray) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) from an (H, W, 3) point cloud registered to the colour
    image: least squares on u = fx * X/Z + cx, v = fy * Y/Z + cy."""
    H, W = point_cloud.shape[:2]
    P = point_cloud.reshape(-1, 3).astype(np.float64)
    v, u = np.divmod(np.arange(H * W), W)
    ok = P[:, 2] > 0
    rng = np.random.default_rng(0)
    idx = np.nonzero(ok)[0]
    idx = idx[rng.choice(len(idx), min(200000, len(idx)), replace=False)]
    xz, yz = P[idx, 0] / P[idx, 2], P[idx, 1] / P[idx, 2]
    fx, cx = np.linalg.lstsq(np.c_[xz, np.ones_like(xz)], u[idx], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[yz, np.ones_like(yz)], v[idx], rcond=None)[0]
    return float(fx), float(fy), float(cx), float(cy)


def rays(u, v, K):
    fx, fy, cx, cy = K
    d = np.stack([(np.asarray(u) - cx) / fx, (np.asarray(v) - cy) / fy, np.ones_like(np.asarray(u, float))], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _bone_lengths(T: np.ndarray, max_dz: float = 350.0) -> dict[tuple[str, str], float]:
    out = {}
    for p, c in BONES:
        A, B = T[:, COCO17.index(p)], T[:, COCO17.index(c)]
        L = np.linalg.norm(A - B, axis=1)
        ok = np.isfinite(L) & (np.abs(A[:, 2] - B[:, 2]) < max_dz) & (L > 100) & (L < 650)
        out[(p, c)] = float(np.median(L[ok])) if ok.sum() >= 30 else np.nan
    # left/right symmetry: fill a missing side from the other
    for (p, c), L in list(out.items()):
        if np.isnan(L):
            other = (p.replace("left", "X").replace("right", "left").replace("X", "right"),
                     c.replace("left", "X").replace("right", "left").replace("X", "right"))
            out[(p, c)] = out.get(other, np.nan)
    return out


def lift_person(T: np.ndarray, uv: np.ndarray, K, *, torso_tol_mm: float = 250.0,
                bone_tol: float = 0.25, scale_hint: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """T: (n, 17, 3) measured mm (NaN = none), uv: (n, 17, 2) pixels (NaN = no 2D).
    Returns corrected (n, 17, 3) and a boolean (n, 17) 'lifted' mask."""
    T = T.copy()
    lifted = np.zeros(T.shape[:2], bool)
    R = rays(uv[..., 0], uv[..., 1], K)                      # (n, 17, 3), NaN where no 2D
    Lb = _bone_lengths(T)
    for (p, c), L in Lb.items():
        if np.isnan(L):
            kind = "upper_arm" if "elbow" in c else "forearm" if "wrist" in c else "thigh" if "knee" in c else "shin"
            Lb[(p, c)] = DEFAULT_BONE_MM[kind] * scale_hint

    # torso: depth outliers back to the torso's median depth, along their ray
    ti = [COCO17.index(k) for k in TORSO]
    zt = np.nanmedian(T[:, ti, 2], axis=1)
    for j in ti:
        bad = (np.abs(T[:, j, 2] - zt) > torso_tol_mm) | (np.isnan(T[:, j, 2]) & np.isfinite(zt))
        bad &= np.isfinite(R[:, j, 0]) & np.isfinite(zt)
        T[bad, j] = R[bad, j] * (zt[bad] / R[bad, j, 2])[:, None]
        lifted[bad, j] = True

    # limbs, parent before child
    for (p, c) in BONES:
        ip, ic, L = COCO17.index(p), COCO17.index(c), Lb[(p, c)]
        P, r, M = T[:, ip], R[:, ic], T[:, ic]
        have = np.isfinite(P[:, 0]) & np.isfinite(r[:, 0])
        meas_ok = np.isfinite(M[:, 0]) & (np.abs(np.linalg.norm(M - P, axis=1) - L) <= bone_tol * L)
        todo = have & ~meas_ok
        if not todo.any():
            continue
        Pp, rr = P[todo], r[todo]
        b = np.sum(rr * Pp, 1)
        disc = b ** 2 - np.sum(Pp * Pp, 1) + L ** 2
        sq = np.sqrt(np.clip(disc, 0, None))
        s1, s2 = b - sq, b + sq
        # Front/back ambiguity of the two roots: pick the one continuing the
        # joint's last trusted position (temporal continuity); with no
        # recent position, the one at the range closest to the parent's.
        # The measured depth failed the bone check and is not used.
        idx = np.nonzero(todo)[0]
        good = np.nonzero(meas_ok)[0]
        good_rng = np.linalg.norm(T[good, ic], axis=1) if len(good) else np.array([])
        for k, f in enumerate(idx):
            ref = np.linalg.norm(Pp[k])
            if len(good):  # nearest trusted measurement of this joint, past or future
                j = np.searchsorted(good, f)
                cand = [c for c in (j - 1, j) if 0 <= c < len(good)]
                c = min(cand, key=lambda c: abs(good[c] - f))
                if abs(good[c] - f) <= 15:
                    ref = good_rng[c]
            T[f, ic] = rr[k] * (s1[k] if abs(s1[k] - ref) <= abs(s2[k] - ref) else s2[k])
            lifted[f, ic] = True

    # head points: at the depth of the shoulders' centre when theirs is off
    zs = np.nanmean(T[:, [COCO17.index("left_shoulder"), COCO17.index("right_shoulder")], 2], axis=1)
    for k in HEAD:
        j = COCO17.index(k)
        bad = (np.isnan(T[:, j, 2]) | (np.abs(T[:, j, 2] - zs) > 300)) & np.isfinite(R[:, j, 0]) & np.isfinite(zs)
        T[bad, j] = R[bad, j] * (zs[bad] / R[bad, j, 2])[:, None]
        lifted[bad, j] = True
    return T, lifted


def lift_dataframe(df3d: pd.DataFrame, K, scale_hint: dict[int, float] | None = None) -> pd.DataFrame:
    """Applies `lift_person` to every person of a keypoints_3d table."""
    out = []
    for pid, g in df3d.groupby("global_person_id"):
        frames = np.sort(g["frame"].unique())
        fi = {f: i for i, f in enumerate(frames)}
        n = len(frames)
        T = np.full((n, 17, 3), np.nan)
        uv = np.full((n, 17, 2), np.nan)
        g = g[g["keypoint_name"].isin(COCO17)]
        i = g["frame"].map(fi).to_numpy()
        k = g["keypoint_name"].map(COCO17.index).to_numpy()
        T[i, k] = g[["X_mm", "Y_mm", "Z_mm"]].to_numpy()
        uv[i, k] = g[["x", "y"]].to_numpy()
        T2, lifted = lift_person(T, uv, K, scale_hint=(scale_hint or {}).get(int(pid), 1.0))
        ii, kk = np.nonzero(np.isfinite(T2[..., 0]))
        out.append(pd.DataFrame({
            "global_person_id": int(pid), "frame": frames[ii], "keypoint_name": np.array(COCO17)[kk],
            "x": uv[ii, kk, 0], "y": uv[ii, kk, 1],
            "X_mm": T2[ii, kk, 0], "Y_mm": T2[ii, kk, 1], "Z_mm": T2[ii, kk, 2], "lifted": lifted[ii, kk]}))
    return pd.concat(out, ignore_index=True)
