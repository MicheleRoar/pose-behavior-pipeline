"""
pose/room_coverage.py
=======================
Combines two Kinect cameras' raw depth point clouds (via the rigid
transform from pose/table_calibration.py) into a shared floor-plan
view, to answer Michele's original question: how much of the room does
each camera see on its own, and how much MORE does the union of both
cameras cover? Answered WITHOUT needing pose or segmentation on
camera_b -- this only needs each camera's own extracted depth point
cloud (table_calibration.py's `extract` step) plus the rigid transform
between them (`register`).

Method
------
1. Estimate the floor plane from camera_a's own point cloud via RANSAC,
   seeded from points in the lower part of the room (large Y in
   Kinect's camera-space convention: X right, Y down, Z forward -- so
   "large Y" = physically low = floor candidates). This is a genuine
   RANSAC fit from real points, not an assumption that the camera is
   perfectly level.
2. Build a 2D floor coordinate system (u, v) within that plane, plus a
   `height` axis (perpendicular distance from the floor) so floor-level
   points can be told apart from walls/ceiling/people/furniture.
3. Transform camera_b's own point cloud into camera_a's frame via the
   saved rigid transform (pose/table_calibration.py's output) -- both
   point clouds then share the same floor coordinate system.
4. Bin each camera's valid points (real depth return, within a
   plausible room-height band above the floor) into floor-plane grid
   cells. A cell counts as "seen" by a camera if ANY valid point from
   that camera's own view lands in it -- a coarse visibility proxy (a
   "shadow" of what each camera can see on the floor), NOT a precise
   per-tile occupancy or line-of-sight measurement. Said explicitly in
   the report, not hidden behind a clean-looking percentage.
5. Report: cells seen by camera_a alone, camera_b alone, their union,
   their overlap, and how much MORE floor area the union covers versus
   camera_a alone (the actual answer to "quanta percentuale di stanza
   in più vediamo aggiungendo camera_b").

Known limitations (real, not hidden):
- The floor plane is fit from camera_a's OWN points only -- if RANSAC
  locks onto a wall or the table instead of the true floor, the whole
  floor coordinate system is wrong. ALWAYS sanity-check the printed
  `n_inliers`/`rmse_mm` (a real flat floor should fit with a low RMSE
  and a large inlier count relative to the candidate pool) before
  trusting the coverage numbers.
- table_calibration.py's `register` transform carries its own residual
  error (see its README section, confirmed ~74mm on 10_individual_12)
  -- this propagates directly into where camera_b's points land on the
  shared floor plan.
- "Seen" is binary per grid cell (any point vs none) -- it does NOT
  weight by how many points, how central to the FOV, or how reliable
  that specific reading was. A single stray point at the very edge of
  a camera's view still marks that cell as "seen".
- Height-band filtering (`--min-height-mm`/`--max-height-mm`) is a
  blunt tool to exclude ceiling/below-floor noise -- it can't
  distinguish a genuinely tall object from noise at the same height.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pose.table_calibration import apply_transform


def _valid_points(point_cloud: np.ndarray) -> np.ndarray:
    """Flattens an (H, W, 3) point cloud to (N, 3), keeping only points
    with a real depth return (Z>0) -- same convention as depth.py and
    table_calibration.py. Never guesses at the missing ones."""
    pts = point_cloud.reshape(-1, 3).astype(np.float64)
    return pts[pts[:, 2] > 0]


def fit_floor_plane(points_xyz: np.ndarray, *, floor_candidate_percentile: float = 70.0,
                     n_iterations: int = 500, inlier_thresh_mm: float = 30.0,
                     rng: np.random.Generator | None = None) -> dict:
    """RANSAC plane fit, seeded from points likely to be floor (points
    in the lower `100 - floor_candidate_percentile`% of the room by Y
    -- Kinect's camera-space Y points down, so large Y = physically
    low). Tries `n_iterations` random 3-point samples from that
    lower-Y subset, keeps the one with the most inliers (within
    `inlier_thresh_mm` of the plane), then refits the final plane via
    SVD on all its inliers (cleaner than a raw 3-point sample).

    Returns a dict: `normal` (unit vector, oriented to point AWAY from
    the floor and UP into the room -- i.e. negative-Y-ish, since +Y is
    down), `point` (the inlier centroid, a point on the plane),
    `n_inliers`, `n_candidates`, `rmse_mm` (inlier residual -- large
    means this probably isn't a real flat surface, don't trust it)."""
    if rng is None:
        rng = np.random.default_rng(0)
    y = points_xyz[:, 1]
    thresh_y = np.percentile(y, floor_candidate_percentile)
    candidates = points_xyz[y >= thresh_y]
    n = len(candidates)
    if n < 3:
        raise ValueError(f"Only {n} floor-candidate points (need >= 3) -- lower "
                          f"floor_candidate_percentile or check the point cloud has real floor coverage.")

    best = None
    for _ in range(n_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = candidates[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue  # degenerate (near-collinear) sample, skip
        normal = normal / norm
        dist = np.abs((candidates - p0) @ normal)
        inlier_mask = dist < inlier_thresh_mm
        n_inliers = int(inlier_mask.sum())
        if best is None or n_inliers > best["n_inliers"]:
            best = {"n_inliers": n_inliers, "inlier_mask": inlier_mask}

    if best is None:
        raise ValueError("RANSAC found no valid plane (every sampled triple was degenerate) -- try more iterations.")

    inliers = candidates[best["inlier_mask"]]
    centroid = inliers.mean(axis=0)
    centered = inliers - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    normal = normal / np.linalg.norm(normal)
    if normal[1] > 0:  # orient to point UP into the room (negative Y, since +Y is down)
        normal = -normal
    rmse_mm = float(np.sqrt(np.mean(((inliers - centroid) @ normal) ** 2)))
    return {"normal": normal, "point": centroid, "n_inliers": int(best["n_inliers"]),
            "n_candidates": n, "rmse_mm": rmse_mm}


def floor_coords(points_xyz: np.ndarray, plane: dict) -> np.ndarray:
    """Projects points into the floor plane's own 2D (u, v) coordinate
    system plus a `height` axis (mm from the floor, along the plane's
    normal -- positive means above the floor). Returns an (N, 3) array
    of (u_mm, v_mm, height_mm)."""
    normal = plane["normal"]
    origin = plane["point"]
    x_hint = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, x_hint)) > 0.9:
        x_hint = np.array([0.0, 0.0, 1.0])
    u = np.cross(normal, x_hint)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    rel = points_xyz - origin
    return np.stack([rel @ u, rel @ v, rel @ normal], axis=1)


def compute_coverage(
    *, cam_a_pointcloud_npy: str, cam_b_pointcloud_npy: str, transform_json: str,
    cell_size_mm: float = 100.0, min_height_mm: float = -150.0, max_height_mm: float = 2200.0,
) -> dict:
    """`transform_json`: table_calibration.py `register`'s output,
    mapping camera_b's points into camera_a's frame. The floor plane is
    fit from camera_a's OWN points (see `fit_floor_plane`) -- always
    check the returned `plane` diagnostics before trusting the result.
    `min_height_mm`/`max_height_mm`: keep only points within this band
    above the fitted floor (default: 15cm below to 2.2m above, a
    generous "plausible room contents" band that excludes obvious
    below-floor noise and ceiling returns)."""
    pts_a = _valid_points(np.load(cam_a_pointcloud_npy))
    pts_b_own = _valid_points(np.load(cam_b_pointcloud_npy))

    transform = json.loads(Path(transform_json).read_text())
    R = np.array(transform["rotation_3x3"])
    t = np.array(transform["translation_mm"])
    pts_b = apply_transform(pts_b_own, R, t)  # now in camera_a's frame

    plane = fit_floor_plane(pts_a)
    fc_a = floor_coords(pts_a, plane)
    fc_b = floor_coords(pts_b, plane)

    def in_band(fc):
        return (fc[:, 2] >= min_height_mm) & (fc[:, 2] <= max_height_mm)

    fc_a = fc_a[in_band(fc_a)]
    fc_b = fc_b[in_band(fc_b)]

    def cell_set(fc):
        cu = np.floor(fc[:, 0] / cell_size_mm).astype(np.int64)
        cv = np.floor(fc[:, 1] / cell_size_mm).astype(np.int64)
        return set(zip(cu.tolist(), cv.tolist()))

    cells_a = cell_set(fc_a)
    cells_b = cell_set(fc_b)
    union = cells_a | cells_b
    inter = cells_a & cells_b
    cell_area_m2 = (cell_size_mm / 1000.0) ** 2

    return {
        "plane": {"normal": plane["normal"].tolist(), "point": plane["point"].tolist(),
                  "n_inliers": plane["n_inliers"], "n_candidates": plane["n_candidates"],
                  "rmse_mm": plane["rmse_mm"]},
        "cell_size_mm": cell_size_mm,
        "n_cells_a": len(cells_a),
        "n_cells_b": len(cells_b),
        "n_cells_union": len(union),
        "n_cells_overlap": len(inter),
        "overlap_fraction_of_union": (len(inter) / len(union)) if union else float("nan"),
        "gain_from_adding_b_fraction": ((len(union) - len(cells_a)) / len(cells_a)) if cells_a else float("nan"),
        "area_m2_a": len(cells_a) * cell_area_m2,
        "area_m2_b": len(cells_b) * cell_area_m2,
        "area_m2_union": len(union) * cell_area_m2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam-a-pointcloud", required=True, help="cam_a_pointcloud.npy from table_calibration.py extract")
    parser.add_argument("--cam-b-pointcloud", required=True, help="cam_b_pointcloud.npy from table_calibration.py extract")
    parser.add_argument("--transform", required=True, help="camera_b_to_camera_a_transform.json from table_calibration.py register")
    parser.add_argument("--cell-size-mm", type=float, default=100.0)
    parser.add_argument("--min-height-mm", type=float, default=-150.0)
    parser.add_argument("--max-height-mm", type=float, default=2200.0)
    parser.add_argument("--out", default=None, help="optional: write the full report as JSON")
    args = parser.parse_args()

    report = compute_coverage(
        cam_a_pointcloud_npy=args.cam_a_pointcloud, cam_b_pointcloud_npy=args.cam_b_pointcloud,
        transform_json=args.transform, cell_size_mm=args.cell_size_mm,
        min_height_mm=args.min_height_mm, max_height_mm=args.max_height_mm,
    )
    p = report["plane"]
    print(f"[room_coverage] floor plane: {p['n_inliers']}/{p['n_candidates']} inliers, "
          f"RMSE={p['rmse_mm']:.1f}mm, normal={[round(x, 3) for x in p['normal']]} "
          f"-- SANITY-CHECK this looks like a real floor (low RMSE, high inlier fraction) before trusting the rest.")
    print(f"[room_coverage] camera_a alone: {report['n_cells_a']} cells (~{report['area_m2_a']:.1f} m^2)")
    print(f"[room_coverage] camera_b alone: {report['n_cells_b']} cells (~{report['area_m2_b']:.1f} m^2)")
    print(f"[room_coverage] union (both cameras): {report['n_cells_union']} cells (~{report['area_m2_union']:.1f} m^2)")
    print(f"[room_coverage] overlap: {report['n_cells_overlap']} cells "
          f"({report['overlap_fraction_of_union'] * 100:.1f}% of the union)")
    print(f"[room_coverage] adding camera_b increases floor coverage by "
          f"{report['gain_from_adding_b_fraction'] * 100:.1f}% over camera_a alone")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[room_coverage] full report -> {args.out}")


if __name__ == "__main__":
    main()
