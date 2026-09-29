"""
pose/table_calibration.py
===========================
Cross-camera registration for sessions filmed by two Kinect cameras at
once (today: only `10_individual_12` has both `camera_a` and
`camera_b`) -- an idea Matthew raised directly, not a workaround we
invented: register `camera_b`'s 3D coordinate frame onto `camera_a`'s
using the shared LEGO table as a common physical reference.

Why this is simpler than classic stereo calibration
------------------------------------------------------
A textbook two-camera calibration (checkerboard + cv2.stereoCalibrate)
solves for the rigid transform between two cameras from 2D pixel
correspondences plus triangulation -- and needs a dedicated calibration
recording. We don't need any of that: `pose/depth.py` already gives us
REAL 3D points from EACH camera independently (its own embedded depth +
factory calibration). So this reduces to 3D-3D rigid point-set
registration (Kabsch/Umeyama, closed-form via SVD) between a handful of
corresponding table points read from each camera's own depth -- and the
table is already in every session, no special calibration video needed.

Pipeline (mirrors the shape of the earlier, now-superseded 2D
`floor_calibration.py` sketch, but grounded in real depth, not a pixel
homography):

  1. extract   -- grab ONE clean capture (color + depth point cloud)
                  from each camera's ORIGINAL .mkv via pyk4a, at a
                  moment when the table is unobstructed. Saves a PNG
                  (to click on) and the point cloud (.npy) per camera.
  2. pick      -- click N corresponding physical points of the table
                  (e.g. its 4 corners) on each camera's saved PNG, IN
                  THE SAME ORDER on both -- interactive, reused from
                  the old floor_calibration.py's click helper.
  3. lookup3d  -- resolves each clicked 2D pixel to a real 3D point
                  (mm) via that camera's own saved point cloud. A
                  clicked point with no valid depth return is NaN --
                  same "no made-up signal" rule as pose/depth.py; if
                  that happens, re-pick a clearer point instead of
                  guessing one. Confirmed on real 10_individual_12 data:
                  valid/invalid depth returns are often speckled pixel-
                  by-pixel near reflective (white table/walls) or IR-
                  absorptive (dark mats/clothing) surfaces -- finer-
                  grained than a mouse click can reliably land on, even
                  when picking (see below) shows the general valid
                  area. `--search-radius-px` snaps an invalid click to
                  the nearest ACTUAL valid pixel within N pixels (still
                  real data, not invented) instead of forcing an exact
                  hit.
  4. register  -- fits the rigid transform (rotation + translation)
                  that best maps one camera's table points onto the
                  other's (closed-form least squares, works from as
                  few as 3 non-collinear correspondences; more points
                  -> a more robust fit). Reports the residual RMS
                  error in mm as a fit-quality signal -- a large RMSE
                  means the clicked points didn't correspond well
                  (wrong order, wrong physical point, or bad depth).
  5. apply     -- applies a saved transform to a camera_b
                  `keypoints_3d.csv` (pose/depth.py's output),
                  producing a copy expressed in camera_a's frame --
                  directly comparable to camera_a's own 3D keypoints.

Not needed for the core per-camera movement-metrics validation
(pose/movement_metrics.py) -- distance/velocity/variability within one
fixed camera's own frame already make sense without this. This step
only matters once we want to cross-check or fuse the two views of the
same session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Grab one clean color+depth capture from a camera's original .mkv
# ---------------------------------------------------------------------------

def extract_reference(
    *,
    mkv_path: str,
    out_prefix: str,
    t_seconds: float = 0.0,
    max_frames_to_scan: int = 300,
) -> dict[str, str]:
    """Seeks to `t_seconds` into `mkv_path` and walks forward (up to
    `max_frames_to_scan` captures) until it finds one with BOTH a
    decodable color frame and a non-None depth point cloud, then saves:
      `<out_prefix>_color.png`       -- decoded BGR frame, to click on
      `<out_prefix>_pointcloud.npy`  -- (H, W, 3) mm point cloud, color-
                                         camera frame, for 3D lookup
    Raises RuntimeError if no such capture is found in the scanned
    window (try a different `t_seconds` -- e.g. avoid the very start of
    the recording, which is sometimes missing a capture or two)."""
    try:
        from pyk4a import PyK4APlayback, SeekOrigin
    except ImportError as exc:
        raise ImportError(
            "pose/table_calibration.py requires 'pyk4a' AND the real Azure Kinect "
            "Sensor SDK (libk4a) -- see README.md's Setup section."
        ) from exc
    import cv2

    playback = PyK4APlayback(str(mkv_path))
    playback.open()
    if t_seconds:
        playback.seek(int(round(t_seconds * 1_000_000)), SeekOrigin.BEGIN)

    found = None
    try:
        for _ in range(max_frames_to_scan):
            try:
                capture = playback.get_next_capture()
            except EOFError:
                break
            if capture.color is None or capture.transformed_depth_point_cloud is None:
                continue
            color_bgr = cv2.imdecode(np.frombuffer(capture.color, dtype=np.uint8), cv2.IMREAD_COLOR)
            if color_bgr is None:
                continue  # color track wasn't actually MJPG-decodable this capture, try the next one
            found = (color_bgr, capture.transformed_depth_point_cloud)
            break
    finally:
        playback.close()

    if found is None:
        raise RuntimeError(
            f"No capture with both a decodable color frame and a valid depth point cloud "
            f"found in the first {max_frames_to_scan} captures from t={t_seconds}s of "
            f"{mkv_path} -- try a different --t."
        )

    color_bgr, point_cloud = found
    color_path = f"{out_prefix}_color.png"
    pc_path = f"{out_prefix}_pointcloud.npy"
    Path(color_path).parent.mkdir(parents=True, exist_ok=True)
    import cv2 as _cv2
    _cv2.imwrite(color_path, color_bgr)
    np.save(pc_path, point_cloud)
    print(f"[table_calibration] color frame -> {color_path}")
    print(f"[table_calibration] depth point cloud ({point_cloud.shape}) -> {pc_path}")
    return {"color": color_path, "pointcloud": pc_path}


# ---------------------------------------------------------------------------
# 2. Interactive point picking (ported from the earlier floor_calibration.py)
# ---------------------------------------------------------------------------

def _ensure_interactive_backend() -> str:
    """matplotlib silently falls back to the headless 'Agg' backend when it
    can't find a GUI toolkit (or there's no display) -- `plt.show()` then
    does NOTHING (no window, no error) and the click loop collects 0
    points, which used to surface as a confusing "Expected 4 points, got
    0" far from the real cause. This tries a short list of common
    interactive backends BEFORE pyplot is imported (switching after import
    is unreliable) and raises a clear, actionable error if none work,
    instead of silently proceeding with Agg. Returns the backend name that
    ended up active."""
    import matplotlib

    for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg"):
        try:
            matplotlib.use(candidate, force=True)
            return candidate
        except Exception:
            continue

    import os
    display_hint = (
        "$DISPLAY is not set -- this session likely has no display at all (headless machine, "
        "or SSH without X11 forwarding / -X). Interactive point-picking needs a real display; "
        "see the fallback below."
        if not os.environ.get("DISPLAY")
        else "$DISPLAY is set, so a display exists, but no GUI toolkit for matplotlib is "
             "installed -- try: sudo apt install python3-tk (then re-run this command)."
    )
    raise RuntimeError(
        f"No interactive matplotlib backend available (tried TkAgg/QtAgg/Qt5Agg/GTK3Agg, all "
        f"failed) -- matplotlib would otherwise silently fall back to 'Agg', which cannot show "
        f"a window at all. {display_hint}\n"
        f"Fallback if no display will ever be available here: open the saved *_color.png in any "
        f"local image viewer, hover over the table corners to read their pixel (x, y), and hand "
        f"those 4 coordinate pairs over directly instead of using 'pick'."
    )


def _click_points(img_bgr: np.ndarray, title: str, n_points: int | None = None,
                   valid_mask: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Opens a window; left click = add point, right click/backspace-like
    = undo last, close window (or reaching n_points) = done.

    `valid_mask`: optional (H, W) bool array, same pixel size as
    `img_bgr`, True where that pixel has a real (non-NaN, Z>0) depth
    return in the camera's own point cloud (see `pick_points`'s
    `pointcloud_path`). When given, the area with NO valid depth is
    shown darkened/red-tinted -- a visual guide so you can pick your 4-6
    points inside the good area to begin with, instead of finding out
    only after `lookup3d` that a click landed outside the depth FOV.
    It's a guide, not a hard block -- you can still click anywhere;
    `lookup3d` will flag an invalid pick as NaN either way."""
    import cv2

    _ensure_interactive_backend()
    import matplotlib.pyplot as plt

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    subtitle = "left click = add point, right click = undo, close window = done"
    if valid_mask is not None:
        if valid_mask.shape[:2] != img_rgb.shape[:2]:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape[:2]} doesn't match image shape {img_rgb.shape[:2]}")
        img_rgb = img_rgb.copy()
        invalid = ~valid_mask
        red = np.array([140, 0, 0], dtype=np.float64)
        img_rgb[invalid] = (img_rgb[invalid].astype(np.float64) * 0.35 + red * 0.65).astype(np.uint8)
        subtitle += "\n(dark red area = no valid depth there -- click inside the normal-colored area)"

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_rgb)
    ax.set_title(f"{title}\n{subtitle}")
    pts: list[tuple[float, float]] = []
    markers = []

    def redraw():
        for m in markers:
            m.remove()
        markers.clear()
        for i, (x, y) in enumerate(pts):
            m, = ax.plot(x, y, "o", color="red", markersize=6)
            markers.append(m)
            t = ax.annotate(str(i), (x, y), color="yellow", fontsize=9,
                             xytext=(4, 4), textcoords="offset points")
            markers.append(t)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            pts.append((event.xdata, event.ydata))
        elif event.button == 3 and pts:
            pts.pop()
        redraw()
        if n_points is not None and len(pts) >= n_points:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return pts


def pick_points(*, color_image_path: str, out_json: str, n_points: int = 4, label: str = "",
                 pointcloud_path: str | None = None) -> str:
    """`pointcloud_path`: optional -- the same camera's `*_pointcloud.npy`
    from `extract`. When given, the click window overlays which pixels
    have a real depth return (see `_click_points`'s `valid_mask`), so you
    can see and avoid the no-depth area up front rather than discovering
    a bad pick only at the `lookup3d` step."""
    import cv2

    img = cv2.imread(str(color_image_path))
    if img is None:
        raise SystemExit(f"Cannot read {color_image_path}")
    valid_mask = None
    if pointcloud_path:
        point_cloud = np.load(pointcloud_path)
        if point_cloud.shape[:2] != img.shape[:2]:
            raise SystemExit(
                f"Point cloud shape {point_cloud.shape[:2]} doesn't match image shape {img.shape[:2]} "
                f"-- are --color-image and --pointcloud from the same 'extract' run?")
        valid_mask = point_cloud[..., 2] > 0
    title = f"Click {n_points} table points{(' -- ' + label) if label else ''} (same physical points, same order, on both cameras)"
    pts = _click_points(img, title, n_points=n_points, valid_mask=valid_mask)
    if len(pts) != n_points:
        raise SystemExit(f"Expected {n_points} points, got {len(pts)}. Re-run and click exactly {n_points}.")
    Path(out_json).write_text(json.dumps({"image": str(color_image_path), "pixel_points": pts}, indent=2))
    print(f"[table_calibration] {n_points} pixel points -> {out_json}")
    return out_json


# ---------------------------------------------------------------------------
# 3. Resolve clicked pixels to real 3D points via the saved point cloud
# ---------------------------------------------------------------------------

def lookup_3d(point_cloud: np.ndarray, pixel_points: list[tuple[float, float]],
              search_radius_px: int = 0) -> np.ndarray:
    """Returns an (N, 3) array of (X_mm, Y_mm, Z_mm) -- NaN row for any
    point outside the point cloud's bounds or with no valid depth
    return (Z<=0), same convention as pose/depth.py. Never guesses.

    `search_radius_px`: confirmed on real 10_individual_12 data --
    valid/invalid depth returns are often speckled pixel-by-pixel near
    reflective (white table, walls) or IR-absorptive (dark mats/
    clothing) surfaces (see module docstring), finer-grained than a
    mouse click (and its on-screen display scaling) can reliably land
    on. When the EXACT clicked pixel is invalid and this is > 0, this
    searches a square neighborhood out to `search_radius_px` pixels
    (Chebyshev) for the nearest (Euclidean) pixel that DOES have valid
    depth, and uses that instead. This is still a real, observed depth
    reading -- not invented -- just tolerant of a click landing 1-few
    pixels into a no-return speckle immediately next to good data. 0
    (default) disables this, matching the original exact-pixel-only
    behavior."""
    ph, pw = point_cloud.shape[:2]
    out = np.full((len(pixel_points), 3), np.nan, dtype=np.float64)
    for i, (x, y) in enumerate(pixel_points):
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < pw and 0 <= py < ph):
            continue
        Xc, Yc, Zc = point_cloud[py, px]
        if Zc > 0:
            out[i] = (float(Xc), float(Yc), float(Zc))
            continue
        if search_radius_px > 0:
            y0, y1 = max(0, py - search_radius_px), min(ph, py + search_radius_px + 1)
            x0, x1 = max(0, px - search_radius_px), min(pw, px + search_radius_px + 1)
            window = point_cloud[y0:y1, x0:x1]
            valid_yx = np.argwhere(window[..., 2] > 0)
            if len(valid_yx) > 0:
                yy = valid_yx[:, 0] + y0
                xx = valid_yx[:, 1] + x0
                d2 = (xx - x) ** 2 + (yy - y) ** 2
                j = int(np.argmin(d2))
                out[i] = (float(window[valid_yx[j, 0], valid_yx[j, 1], 0]),
                          float(window[valid_yx[j, 0], valid_yx[j, 1], 1]),
                          float(window[valid_yx[j, 0], valid_yx[j, 1], 2]))
    return out


def lookup_3d_from_files(*, pointcloud_npy: str, points_json: str, out_json: str,
                          search_radius_px: int = 0) -> str:
    point_cloud = np.load(pointcloud_npy)
    pixel_points = json.loads(Path(points_json).read_text())["pixel_points"]
    xyz = lookup_3d(point_cloud, pixel_points, search_radius_px=search_radius_px)
    n_invalid = int(np.isnan(xyz).any(axis=1).sum())
    if n_invalid:
        hint = (" (already tried snapping to the nearest valid pixel within "
                 f"{search_radius_px}px -- try a larger --search-radius-px, or re-pick "
                 "those points on a clearer spot)") if search_radius_px > 0 else \
               " -- try re-picking with a larger --search-radius-px (e.g. 5-10) instead of an exact click"
        print(f"[table_calibration] WARNING: {n_invalid}/{len(pixel_points)} clicked points have no "
              f"valid depth (NaN){hint}, a registration needs at least 3 valid, non-collinear points.")
    Path(out_json).write_text(json.dumps({"points_3d_mm": xyz.tolist()}, indent=2))
    print(f"[table_calibration] {len(pixel_points) - n_invalid}/{len(pixel_points)} valid 3D points -> {out_json}")
    return out_json


# ---------------------------------------------------------------------------
# 4. Rigid transform (Kabsch/Umeyama) between two corresponding point sets
# ---------------------------------------------------------------------------

def compute_rigid_transform(
    source_xyz: np.ndarray, target_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Closed-form least-squares rigid transform (R, t) minimizing
    sum(||R @ source_i + t - target_i||^2), via SVD (Kabsch/Umeyama).
    Rows that are NaN in EITHER array (unresolved depth at pick time)
    are dropped before fitting. Returns (R (3,3), t (3,), rmse_mm, n_used)
    -- rmse_mm is the residual fit error, a large value (much more than
    a few mm) means the correspondences don't actually line up (wrong
    click order, wrong physical points, or bad depth reads) and the
    registration shouldn't be trusted. Raises ValueError if fewer than
    3 valid (non-NaN) correspondences remain."""
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    if source_xyz.shape != target_xyz.shape or source_xyz.shape[1] != 3:
        raise ValueError(f"Expected two (N,3) arrays of the same shape, got {source_xyz.shape} vs {target_xyz.shape}")

    valid = ~(np.isnan(source_xyz).any(axis=1) | np.isnan(target_xyz).any(axis=1))
    n_used = int(valid.sum())
    if n_used < 3:
        raise ValueError(f"Need at least 3 valid (non-NaN) point correspondences, got {n_used}")

    S = source_xyz[valid]
    T = target_xyz[valid]

    centroid_s = S.mean(axis=0)
    centroid_t = T.mean(axis=0)
    Sc = S - centroid_s
    Tc = T - centroid_t

    H = Sc.T @ Tc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = centroid_t - R @ centroid_s

    predicted = (R @ S.T).T + t
    rmse_mm = float(np.sqrt(np.mean(np.sum((predicted - T) ** 2, axis=1))))
    return R, t, rmse_mm, n_used


def per_point_residuals_mm(source_xyz: np.ndarray, target_xyz: np.ndarray,
                            R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-correspondence residual distance (mm) after applying the
    already-fit (R, t) to `source_xyz` -- NaN for any row that was NaN
    in either input (so it wasn't used in the fit either). The overall
    `rmse_mm` from `compute_rigid_transform` is the RMS of exactly these
    values -- this breaks it out per point so you can see WHICH
    correspondence(s) are dragging up a bad RMSE (e.g. one point
    clicked in the wrong order, or on the wrong physical spot) instead
    of just an aggregate number that doesn't say where the problem is."""
    predicted = apply_transform(source_xyz, R, t)
    return np.sqrt(np.sum((predicted - np.asarray(target_xyz, dtype=np.float64)) ** 2, axis=1))


def best_cyclic_alignment(source_xyz: np.ndarray, target_xyz: np.ndarray) -> dict:
    """When corresponding points were clicked around a shared physical
    boundary (e.g. the table's corners) on two cameras, a very easy,
    common mistake is starting from a different corner and/or picking
    the opposite rotational direction on the two cameras -- especially
    if the two cameras face each other from roughly opposite sides of
    the table: a "clockwise" order in one camera's image is the SAME
    physical rotation as a "counter-clockwise" order in the other's
    (like reading a clock from the front vs. from behind it). This
    produces exactly the diffuse, moderately-elevated-everywhere
    residual pattern (no single extreme outlier) that a genuinely wrong
    single point would NOT produce.

    This tries every cyclic shift of `source_xyz`'s point order, in
    both the original and reversed direction (2*N relabelings for N
    points), fits a rigid transform for each against `target_xyz`
    (unchanged), and returns the best (lowest RMSE) candidate. This
    does NOT invent or discard any clicked point -- it only tries
    relabelings of the SAME points you already clicked, and reports
    exactly which relabeling won so you can sanity-check it makes
    physical sense (e.g. "camera B's point 0 is actually camera A's
    point 2") rather than blindly trusting an automatic fix."""
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    n = source_xyz.shape[0]
    base = list(range(n))
    best = None
    for reversed_flag, seq in ((False, base), (True, base[::-1])):
        for shift in range(n):
            order = seq[shift:] + seq[:shift]
            try:
                R, t, rmse_mm, n_used = compute_rigid_transform(source_xyz[order], target_xyz)
            except ValueError:
                continue
            if best is None or rmse_mm < best["rmse_mm"]:
                best = {"reversed": reversed_flag, "shift": shift, "source_order": order,
                        "R": R, "t": t, "rmse_mm": rmse_mm, "n_used": n_used}
    if best is None:
        raise ValueError("No candidate relabeling had enough valid (non-NaN) correspondences to fit.")
    return best


def apply_transform(points_xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Applies (R, t) to an (N,3) array -- NaN rows pass through as NaN."""
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    out = np.full_like(points_xyz, np.nan)
    valid = ~np.isnan(points_xyz).any(axis=1)
    out[valid] = (R @ points_xyz[valid].T).T + t
    return out


def register(*, source_points3d_json: str, target_points3d_json: str, out_json: str,
             try_cyclic_shifts: bool = False) -> str:
    """`source` -> the camera whose points will be transformed (e.g.
    camera_b), `target` -> the camera whose frame we're registering
    INTO (e.g. camera_a). Saves the transform + fit diagnostics.

    `try_cyclic_shifts`: see `best_cyclic_alignment` -- if the naive fit
    (points matched 1:1 in click order) has a high RMSE, also tries
    every cyclic shift / reversed-direction relabeling of `source`'s
    points and uses whichever fit is best, IF it's a real improvement.
    Always reports both numbers and the winning relabeling -- this
    tries relabelings of the SAME real clicked points, it never invents
    a point, but you should still sanity-check the reported relabeling
    against what you actually clicked before trusting it."""
    source_xyz = np.array(json.loads(Path(source_points3d_json).read_text())["points_3d_mm"])
    target_xyz = np.array(json.loads(Path(target_points3d_json).read_text())["points_3d_mm"])
    R, t, rmse_mm, n_used = compute_rigid_transform(source_xyz, target_xyz)
    print(f"[table_calibration] naive fit (points matched in click order): {n_used}/{source_xyz.shape[0]} "
          f"valid point(s), residual RMSE = {rmse_mm:.1f} mm")

    used_relabeling = None
    if try_cyclic_shifts and rmse_mm > 30.0:
        best = best_cyclic_alignment(source_xyz, target_xyz)
        if best["rmse_mm"] < rmse_mm - 1.0:  # a real improvement, not just fit noise
            print(f"[table_calibration] found a better relabeling: source point order "
                  f"{best['source_order']}{' (reversed direction)' if best['reversed'] else ''} "
                  f"-> RMSE {best['rmse_mm']:.1f} mm (vs {rmse_mm:.1f} mm naive). This means "
                  f"'source point {best['source_order']}[i]' actually corresponds to 'target point i' "
                  f"-- sanity-check this against what you clicked (does it make sense given the two "
                  f"camera views?) before trusting it.")
            R, t, rmse_mm, n_used = best["R"], best["t"], best["rmse_mm"], best["n_used"]
            used_relabeling = {"reversed": best["reversed"], "source_order": best["source_order"]}
        else:
            print(f"[table_calibration] tried cyclic-shift/reversed relabelings, best found was "
                  f"{best['rmse_mm']:.1f} mm -- no meaningfully better than the naive {rmse_mm:.1f} mm, "
                  f"so the naive fit is kept. The high RMSE likely isn't a simple order/direction mismatch.")

    Path(out_json).write_text(json.dumps({
        "source": str(source_points3d_json),
        "target": str(target_points3d_json),
        "rotation_3x3": R.tolist(),
        "translation_mm": t.tolist(),
        "rmse_mm": rmse_mm,
        "n_points_used": n_used,
        "n_points_total": int(source_xyz.shape[0]),
        "relabeling_applied": used_relabeling,
    }, indent=2))
    print(f"[table_calibration] final: fit from {n_used}/{source_xyz.shape[0]} valid point(s), "
          f"residual RMSE = {rmse_mm:.1f} mm -> {out_json}")
    if rmse_mm > 30.0:
        print(f"[table_calibration] WARNING: {rmse_mm:.1f} mm residual is large for a rigid table "
              f"surface -- check the clicked points are the SAME physical corners, in the SAME order, "
              f"on both cameras, before trusting this transform.")
        residuals = per_point_residuals_mm(
            source_xyz[used_relabeling["source_order"]] if used_relabeling else source_xyz, target_xyz, R, t)
        print("[table_calibration] per-point residuals (mm) -- one or two large outliers among "
              "otherwise-small values usually means THOSE points were clicked out of order or on the "
              "wrong physical spot; uniformly large values across all points usually means the whole "
              "click order is offset/mirrored between the two cameras:")
        for i, r in enumerate(residuals):
            tag = "N/A (invalid depth)" if np.isnan(r) else f"{r:.1f} mm"
            print(f"  point {i}: {tag}")
    return out_json


# ---------------------------------------------------------------------------
# 5. Apply a saved transform to a pose/depth.py keypoints_3d.csv
# ---------------------------------------------------------------------------

def apply_to_csv(*, keypoints_3d_csv: str, transform_json: str, out_csv: str) -> str:
    import pandas as pd

    transform = json.loads(Path(transform_json).read_text())
    R = np.array(transform["rotation_3x3"])
    t = np.array(transform["translation_mm"])

    df = pd.read_csv(keypoints_3d_csv)
    xyz = df[["X_mm", "Y_mm", "Z_mm"]].to_numpy()
    df[["X_mm", "Y_mm", "Z_mm"]] = apply_transform(xyz, R, t)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[table_calibration] {keypoints_3d_csv} transformed into the target camera's frame -> {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="grab one clean color+depth capture from a camera's original .mkv")
    pe.add_argument("--video", required=True, help="ORIGINAL .mkv for ONE camera")
    pe.add_argument("--out-prefix", required=True, help="e.g. cam_a  ->  cam_a_color.png, cam_a_pointcloud.npy")
    pe.add_argument("--t", type=float, default=0.0, help="seconds into the recording to seek to first")
    pe.set_defaults(func=lambda a: extract_reference(mkv_path=a.video, out_prefix=a.out_prefix, t_seconds=a.t))

    pp = sub.add_parser("pick", help="click N corresponding table points on a saved color frame")
    pp.add_argument("--color-image", required=True)
    pp.add_argument("--out", required=True, help="e.g. cam_a_points.json")
    pp.add_argument("--n-points", type=int, default=4)
    pp.add_argument("--label", default="", help="e.g. 'camera A' -- shown in the window title only")
    pp.add_argument("--pointcloud", default=None,
                     help="optional: the same camera's *_pointcloud.npy from 'extract' -- overlays the "
                          "no-valid-depth area on the click window so you can avoid it up front")
    pp.set_defaults(func=lambda a: pick_points(color_image_path=a.color_image, out_json=a.out, n_points=a.n_points,
                                                label=a.label, pointcloud_path=a.pointcloud))

    pl = sub.add_parser("lookup3d", help="resolve clicked pixels to real 3D points via the saved point cloud")
    pl.add_argument("--pointcloud", required=True, help="the *_pointcloud.npy from 'extract'")
    pl.add_argument("--points", required=True, help="the *_points.json from 'pick'")
    pl.add_argument("--out", required=True, help="e.g. cam_a_points3d.json")
    pl.add_argument("--search-radius-px", type=int, default=0,
                     help="if a clicked pixel has no valid depth, search this many pixels around it for the "
                          "nearest valid one instead (real data, just tolerant of imprecise clicks on speckled "
                          "returns -- see lookup_3d's docstring). 0 (default) = exact pixel only")
    pl.set_defaults(func=lambda a: lookup_3d_from_files(pointcloud_npy=a.pointcloud, points_json=a.points,
                                                         out_json=a.out, search_radius_px=a.search_radius_px))

    pr = sub.add_parser("register", help="fit the rigid transform mapping source camera's points onto target camera's")
    pr.add_argument("--source", required=True, help="*_points3d.json for the camera to transform (e.g. camera_b)")
    pr.add_argument("--target", required=True, help="*_points3d.json for the camera to register into (e.g. camera_a)")
    pr.add_argument("--out", required=True, help="e.g. camera_b_to_camera_a_transform.json")
    pr.add_argument("--try-cyclic-shifts", action="store_true",
                     help="if the naive (in-click-order) fit has a high RMSE, also try every cyclic-shift/"
                          "reversed-direction relabeling of the source points and use it if clearly better "
                          "-- see best_cyclic_alignment's docstring (handles the 'different starting corner "
                          "or mirrored rotation direction between two facing cameras' mistake)")
    pr.set_defaults(func=lambda a: register(source_points3d_json=a.source, target_points3d_json=a.target,
                                             out_json=a.out, try_cyclic_shifts=a.try_cyclic_shifts))

    pa = sub.add_parser("apply", help="apply a saved transform to a keypoints_3d.csv")
    pa.add_argument("--keypoints-3d-csv", required=True)
    pa.add_argument("--transform", required=True)
    pa.add_argument("--out", required=True)
    pa.set_defaults(func=lambda a: apply_to_csv(keypoints_3d_csv=a.keypoints_3d_csv, transform_json=a.transform, out_csv=a.out))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
