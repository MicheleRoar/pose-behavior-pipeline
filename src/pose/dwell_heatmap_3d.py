"""
pose/dwell_heatmap_3d.py
==========================
Where in the room each person STAYS (not just passes through): a dwell
heatmap on the floor, per person and per role (e.g. child vs therapist),
drawn inside the room's own 3D point cloud so the table, walls and
furniture are visible around it.

Inputs
-------
- camera_a `keypoints_3d.csv` (camera_a's own frame).
- Optionally camera_b `keypoints_3d.csv` + the alignment JSON from
  `pose.align_cameras_from_people` (its `rotation_3x3`/`translation_mm`,
  `assignment` and `frame_offset_b_to_a`): camera_b's people are moved
  into camera_a's frame and merged with their camera_a counterpart, so a
  person hidden from one camera is still counted from the other.
- camera_a's `*_pointcloud.npy` (`table_calibration extract`): the floor
  plane is fitted from it (`room_coverage.fit_floor_plane`) and a
  subsample of it is drawn as the room.
- `--roles 0:therapist,1:therapist,2:child,3:parent` -- camera_a ids.

What counts as "staying"
-------------------------
Per person, a robust torso position per frame (median of shoulders and
hips, same as `pose.align_cameras_from_people`) goes through
`pose.movement_metrics.compute_movement_metrics` -- same outlier
rejection, smoothing and stillness deadzone already validated on this
session. A frame is "still" when its velocity is zeroed by the deadzone;
only runs of still frames lasting at least `--min-dwell-s` seconds count
(short gaps of up to `--max-gap-frames` inside a run are bridged), so a
person briefly pausing while walking does not light up the map.

Output (in --out-dir)
----------------------
- `dwell_cells.csv`: floor cell (u, v in mm), person/role, dwell seconds.
- `dwell_episodes.csv`: each stay: person, start/end frame, duration,
  mean floor position, mean torso height (sitting vs standing).
- `dwell_heatmap_3d.html`: interactive 3D view (plotly.js from a CDN, no
  Python dependency): room point cloud in grey, one dwell surface per
  role on the floor, toggled from the legend.
- `dwell_heatmap_2d.png`: the same, top view, one panel per role.

Usage:
    cd src
    python -m pose.dwell_heatmap_3d \\
        --camera-a-csv .../pose/camera_a/keypoints_3d.csv \\
        --camera-b-csv .../pose/camera_b/keypoints_3d.csv \\
        --alignment-json .../calib/camera_b_to_camera_a_people.json \\
        --cam-a-pointcloud .../calib/cam_a_pointcloud.npy \\
        --roles 0:therapist,1:therapist,2:child,3:parent \\
        --out-dir .../pose/dwell
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pose.align_cameras_from_people import torso_roots

ROLE_COLORS = {"child": "#e4572e", "therapist": "#2e86ab", "parent": "#7a9e3b"}
DEFAULT_COLOR = "#8e6c8a"


# ----------------------------------------------------------------------------
# Positions
# ----------------------------------------------------------------------------

def fused_roots(df_a: pd.DataFrame, df_b: pd.DataFrame | None, alignment: dict | None) -> pd.DataFrame:
    """Per (camera_a id, frame) torso position in camera_a's frame. With
    camera_b, each matched camera_b person is moved into camera_a's frame
    (and time) and averaged with camera_a where both see the person;
    where only one camera does, that one is used. Column `source` says
    which (a / b / ab)."""
    ra = torso_roots(df_a).rename(columns={"X": "Xa", "Y": "Ya", "Z": "Za"})
    if df_b is None or alignment is None:
        out = ra.rename(columns={"Xa": "X", "Ya": "Y", "Za": "Z"})
        out["source"] = "a"
        return out
    R = np.asarray(alignment["rotation_3x3"], float)
    t = np.asarray(alignment["translation_mm"], float)
    off = int(alignment.get("frame_offset_b_to_a", 0))
    b_to_a = {int(d["id_b"]): int(d["id_a"]) for d in alignment["assignment"]}
    rb = torso_roots(df_b)
    rb = rb[rb["global_person_id"].isin(b_to_a)].copy()
    xyz = rb[["X", "Y", "Z"]].to_numpy() @ R.T + t
    rb[["Xb", "Yb", "Zb"]] = xyz
    rb["global_person_id"] = rb["global_person_id"].map(b_to_a)
    rb["frame"] = rb["frame"] + off
    rb = rb[["global_person_id", "frame", "Xb", "Yb", "Zb"]]
    m = ra.merge(rb, on=["global_person_id", "frame"], how="outer")
    has_a, has_b = m["Xa"].notna(), m["Xb"].notna()
    for ax in "XYZ":
        m[ax] = m[[f"{ax}a", f"{ax}b"]].mean(axis=1)
    m["source"] = np.where(has_a & has_b, "ab", np.where(has_a, "a", "b"))
    return m[["global_person_id", "frame", "X", "Y", "Z", "source"]]


def movement_from_roots(roots: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Runs the validated movement_metrics chain on the torso roots (fed
    in as both hips, so its hip-midpoint 'root' is exactly this point)."""
    from pose.movement_metrics import compute_movement_metrics

    rows = []
    for k in ("left_hip", "right_hip"):
        x = roots.rename(columns={"X": "X_mm", "Y": "Y_mm", "Z": "Z_mm"}).copy()
        x["keypoint_name"] = k
        x["session_id"] = "fused"
        rows.append(x[["session_id", "global_person_id", "frame", "keypoint_name", "X_mm", "Y_mm", "Z_mm"]])
    per_frame, _ = compute_movement_metrics(pd.concat(rows), fps=fps)
    return per_frame


# ----------------------------------------------------------------------------
# Stays
# ----------------------------------------------------------------------------

def dwell_episodes(per_frame: pd.DataFrame, *, fps: float, min_dwell_s: float, max_gap_frames: int) -> pd.DataFrame:
    """Runs of still frames (deadzone-zeroed velocity) >= min_dwell_s."""
    eps = []
    for pid, g in per_frame.groupby("global_person_id"):
        g = g.sort_values("frame")
        still = g[(g["velocity_mm_s"] == 0) & g["X_mm"].notna()]
        if still.empty:
            continue
        f = still["frame"].to_numpy()
        breaks = np.where(np.diff(f) > max_gap_frames + 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(f) - 1]
        for s, e in zip(starts, ends):
            dur = (f[e] - f[s] + 1) / fps
            if dur < min_dwell_s:
                continue
            seg = still.iloc[s:e + 1]
            eps.append({"global_person_id": int(pid), "start_frame": int(f[s]), "end_frame": int(f[e]),
                        "duration_s": float(dur), "n_still_frames": int(len(seg)),
                        "X_mm": seg["X_mm"].median(), "Y_mm": seg["Y_mm"].median(), "Z_mm": seg["Z_mm"].median(),
                        "frames": seg["frame"].to_numpy(),
                        "xyz": seg[["X_mm", "Y_mm", "Z_mm"]].to_numpy()})
    return pd.DataFrame(eps)


# ----------------------------------------------------------------------------
# Floor + grid
# ----------------------------------------------------------------------------

def floor_frame(plane: dict):
    """Same (u, v, height) axes as room_coverage.floor_coords."""
    from pose.room_coverage import floor_coords
    return lambda pts: floor_coords(np.asarray(pts, float), plane)


def gaussian_blur(grid: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0:
        return grid
    r = int(np.ceil(3 * sigma_cells))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, grid)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)


def dwell_grids(episodes: pd.DataFrame, to_floor, roles: dict[int, str], *, fps: float,
                cell_mm: float, extent: tuple[float, float, float, float]):
    """Seconds of stay per floor cell, per role and per person."""
    u0, u1, v0, v1 = extent
    nu, nv = int(np.ceil((u1 - u0) / cell_mm)), int(np.ceil((v1 - v0) / cell_mm))
    grids, rows = {}, []
    for _, ep in episodes.iterrows():
        pid = int(ep["global_person_id"])
        uvh = to_floor(ep["xyz"])
        iu = np.clip(((uvh[:, 0] - u0) // cell_mm).astype(int), 0, nu - 1)
        iv = np.clip(((uvh[:, 1] - v0) // cell_mm).astype(int), 0, nv - 1)
        for key in (roles.get(pid, f"id{pid}"), f"id{pid}"):
            g = grids.setdefault(key, np.zeros((nu, nv)))
            np.add.at(g, (iu, iv), 1.0 / fps)
    for key, g in grids.items():
        for iu, iv in zip(*np.nonzero(g)):
            rows.append({"group": key, "u_mm": u0 + (iu + 0.5) * cell_mm, "v_mm": v0 + (iv + 0.5) * cell_mm,
                         "dwell_s": float(g[iu, iv])})
    return grids, pd.DataFrame(rows), (nu, nv)


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def render_html(path: Path, *, room_uvh: np.ndarray, groups: list[str], grids: dict, extent, cell_mm: float,
                sigma_cells: float, episodes_uvh: dict, title: str) -> None:
    u0, u1, v0, v1 = extent
    traces = [{
        "type": "scatter3d", "mode": "markers", "name": "room",
        "x": np.round(room_uvh[:, 0] / 1000, 3).tolist(), "y": np.round(room_uvh[:, 1] / 1000, 3).tolist(),
        "z": np.round(room_uvh[:, 2] / 1000, 3).tolist(),
        "marker": {"size": 1.2, "color": np.clip(room_uvh[:, 2] / 2000, 0, 1).tolist(),
                   "colorscale": [[0, "#d9d9d9"], [1, "#6b6b6b"]], "opacity": 0.35},
        "hoverinfo": "skip",
    }]
    vmax = max(gaussian_blur(grids[g], sigma_cells).max() for g in groups) or 1.0
    for i, g in enumerate(groups):
        color = ROLE_COLORS.get(g, DEFAULT_COLOR)
        z = gaussian_blur(grids[g], sigma_cells)
        u = (u0 + (np.arange(z.shape[0]) + 0.5) * cell_mm) / 1000
        v = (v0 + (np.arange(z.shape[1]) + 0.5) * cell_mm) / 1000
        zz = np.where(z > 0.02 * vmax, z, np.nan)
        # surface rises above the floor proportionally to dwell time (max 1 m),
        # so roles can be compared at a glance; real seconds on hover
        traces.append({
            "type": "surface", "name": g, "showlegend": True, "showscale": False,
            "x": u.tolist(), "y": v.tolist(),
            "z": np.where(np.isnan(zz), None, 0.02 + zz / vmax * 1.0).T.tolist(),
            "surfacecolor": np.where(np.isnan(zz), None, zz).T.tolist(),
            "customdata": np.round(z, 1).T.tolist(),
            "colorscale": [[0, "rgba(255,255,255,0)"], [1, color]], "cmin": 0, "cmax": vmax,
            "opacity": 0.85, "hovertemplate": g + ": %{customdata} s<extra></extra>",
        })
        e = episodes_uvh.get(g)
        if e is not None and len(e):
            traces.append({
                "type": "scatter3d", "mode": "markers", "name": f"{g} — stays", "legendgroup": g,
                "x": (e[:, 0] / 1000).tolist(), "y": (e[:, 1] / 1000).tolist(), "z": (e[:, 2] / 1000).tolist(),
                "marker": {"size": np.clip(3 + np.sqrt(e[:, 3]), 3, 18).tolist(), "color": color,
                           "line": {"width": 1, "color": "#ffffff"}},
                "customdata": np.round(e[:, 3], 0).tolist(),
                "hovertemplate": g + ": stay of %{customdata} s, torso at %{z:.2f} m<extra></extra>",
            })
    layout = {
        "title": {"text": title}, "paper_bgcolor": "#ffffff",
        "scene": {"aspectmode": "data",
                  "xaxis": {"title": "u (m)"}, "yaxis": {"title": "v (m)"}, "zaxis": {"title": "height (m)"},
                  "camera": {"eye": {"x": 0.0, "y": -1.6, "z": 1.6}}},
        "legend": {"itemclick": "toggle"}, "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
    }
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>html,body{{margin:0;height:100%;background:#fff;font-family:sans-serif}}#p{{width:100%;height:100%}}</style>
</head><body><div id="p"></div><script>
Plotly.newPlot('p', {json.dumps(traces)}, {json.dumps(layout)}, {{responsive:true}});
</script></body></html>"""
    path.write_text(html)


def render_png(path: Path, *, groups, grids, extent, cell_mm, sigma_cells, room_uvh, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    u0, u1, v0, v1 = extent
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    furniture = room_uvh[(room_uvh[:, 2] > 300) & (room_uvh[:, 2] < 1100)]
    for ax, g in zip(axes[0], groups):
        cmap = LinearSegmentedColormap.from_list(g, ["#ffffff", ROLE_COLORS.get(g, DEFAULT_COLOR)])
        z = gaussian_blur(grids[g], sigma_cells)
        vmax = z.max() or 1.0  # own scale per panel: a person who stays little is still readable
        ax.scatter(furniture[:, 0] / 1000, furniture[:, 1] / 1000, s=0.2, c="#bbbbbb")
        im = ax.imshow(np.ma.masked_less(z.T, 0.02 * vmax), origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                       extent=(u0 / 1000, u1 / 1000, v0 / 1000, v1 / 1000), alpha=0.9)
        ax.set_title(f"{g} — {grids[g].sum() / 60:.1f} min of stays")
        ax.set_aspect("equal"); ax.set_xlabel("u (m)"); ax.set_ylabel("v (m)")
        fig.colorbar(im, ax=ax, shrink=0.7, label="seconds per cell")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_roles(s: str | None) -> dict[int, str]:
    if not s:
        return {}
    return {int(k): v.strip() for k, v in (p.split(":") for p in s.split(","))}


def main() -> None:
    p = argparse.ArgumentParser(description="Dwell heatmap (where each person stays), 3D room view -- see docstring.")
    p.add_argument("--camera-a-csv", required=True)
    p.add_argument("--camera-b-csv", default=None)
    p.add_argument("--alignment-json", default=None, help="output of pose.align_cameras_from_people")
    p.add_argument("--cam-a-pointcloud", required=True, help="camera_a *_pointcloud.npy (floor + room drawing)")
    p.add_argument("--roles", default=None, help="camera_a id:role list, e.g. 0:therapist,1:therapist,2:child,3:parent")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--min-dwell-s", type=float, default=3.0, help="shortest stay that counts")
    p.add_argument("--max-gap-frames", type=int, default=8, help="non-still frames bridged inside one stay")
    p.add_argument("--cell-mm", type=float, default=100.0)
    p.add_argument("--sigma-cells", type=float, default=1.5, help="display smoothing (cells); CSV is unsmoothed")
    p.add_argument("--room-points", type=int, default=40000, help="point-cloud points drawn")
    p.add_argument("--title", default="Where people stay")
    a = p.parse_args()

    from pose.room_coverage import _valid_points, fit_floor_plane

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    roles = parse_roles(a.roles)

    pc = _valid_points(np.load(a.cam_a_pointcloud))
    plane = fit_floor_plane(pc)
    print(f"[dwell] floor: rmse {plane['rmse_mm']:.1f} mm, {plane['n_inliers']} inliers")
    to_floor = floor_frame(plane)

    df_a = pd.read_csv(a.camera_a_csv)
    df_b = pd.read_csv(a.camera_b_csv) if a.camera_b_csv else None
    alignment = json.loads(Path(a.alignment_json).read_text()) if a.alignment_json else None
    roots = fused_roots(df_a, df_b, alignment)
    print("[dwell] frames per source:", roots["source"].value_counts().to_dict())

    per_frame = movement_from_roots(roots[["global_person_id", "frame", "X", "Y", "Z"]], a.fps)
    eps = dwell_episodes(per_frame, fps=a.fps, min_dwell_s=a.min_dwell_s, max_gap_frames=a.max_gap_frames)
    if eps.empty:
        raise SystemExit("No stays found -- lower --min-dwell-s?")

    room_uvh = to_floor(pc[np.random.default_rng(0).choice(len(pc), min(a.room_points, len(pc)), replace=False)])
    room_uvh = room_uvh[(room_uvh[:, 2] > -100) & (room_uvh[:, 2] < 2500)]
    all_uvh = to_floor(np.vstack(eps["xyz"].to_list()))
    lo = np.minimum(np.percentile(room_uvh[:, :2], 1, axis=0), all_uvh[:, :2].min(0)) - 200
    hi = np.maximum(np.percentile(room_uvh[:, :2], 99, axis=0), all_uvh[:, :2].max(0)) + 200
    extent = (lo[0], hi[0], lo[1], hi[1])
    room_uvh = room_uvh[(room_uvh[:, 0] >= lo[0]) & (room_uvh[:, 0] <= hi[0]) &
                        (room_uvh[:, 1] >= lo[1]) & (room_uvh[:, 1] <= hi[1])]

    grids, cells, _ = dwell_grids(eps, to_floor, roles, fps=a.fps, cell_mm=a.cell_mm, extent=extent)
    cells.to_csv(out / "dwell_cells.csv", index=False)

    ep_uvh = to_floor(eps[["X_mm", "Y_mm", "Z_mm"]].to_numpy())
    ep_tab = eps.drop(columns=["frames", "xyz"]).assign(
        role=eps["global_person_id"].map(lambda i: roles.get(int(i), f"id{int(i)}")),
        u_mm=ep_uvh[:, 0], v_mm=ep_uvh[:, 1], torso_height_mm=ep_uvh[:, 2])
    ep_tab.to_csv(out / "dwell_episodes.csv", index=False)

    groups = sorted({roles.get(int(i), f"id{int(i)}") for i in eps["global_person_id"]},
                    key=lambda g: ["child", "therapist", "parent"].index(g) if g in ROLE_COLORS else 9)
    episodes_uvh = {g: np.c_[ep_tab.loc[ep_tab.role == g, ["u_mm", "v_mm", "torso_height_mm", "duration_s"]].to_numpy()]
                    for g in groups}
    render_html(out / "dwell_heatmap_3d.html", room_uvh=room_uvh, groups=groups, grids=grids, extent=extent,
                cell_mm=a.cell_mm, sigma_cells=a.sigma_cells, episodes_uvh=episodes_uvh, title=a.title)
    render_png(out / "dwell_heatmap_2d.png", groups=groups, grids=grids, extent=extent, cell_mm=a.cell_mm,
               sigma_cells=a.sigma_cells, room_uvh=room_uvh, title=a.title)

    summ = ep_tab.groupby("role").agg(stays=("duration_s", "size"), minutes=("duration_s", lambda s: s.sum() / 60),
                                       longest_s=("duration_s", "max"))
    print(summ.round(1).to_string())
    print(f"[dwell] wrote {out}")


if __name__ == "__main__":
    main()
