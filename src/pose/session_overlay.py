"""
pose/session_overlay.py
=========================
One annotated video per camera for checking the whole chain at a glance:

  - a box around each tracked person (from its merged mask, largest blob):
    RED = adult, BLUE = child, GREY = uncertain;
  - the person's ViTPose skeleton (only joints inside its own mask, so the
    drawn skeleton is what was actually seen);
  - a label per person: id, role, CURRENT speed (m/s) and TOTAL distance
    walked so far (m), from the 3D torso position;
  - the posture (standing / seated / on the floor, `pose.posture_3d`),
    also exported as an ELAN tab-delimited file to compare with the codes;
  - an inset map of the room seen from above (floor plane fitted on this
    camera's own depth), filling up over time with where people have been:
    blue = child, red = adults, current positions as dots.

Also writes `occupancy_<camera>.png` (the final map) and `movement_<camera>.csv`
(per person per frame: floor position, speed, cumulative distance).

Inputs, all from one session folder (`--session-dir`):
  processed/<camera>.mp4          full-frame video the masks refer to
  merged/<camera>/<id>.mp4        merged masks (full frame)
  <camera>.mkv                    original Kinect recording (floor plane)
  pose/<camera>_vitpose/          keypoints_smoothed.csv + keypoints_3d.csv
                                  (from `pose.run_pose_cropped --backend vitpose`)
  roles: `--roles 0:adult,...` or the age_group.csv in the pose folder
  (`pose.classify_age_group`).

Speed and distance come from the robust torso position (median of the
shoulders and hips that have a depth reading, as in
`pose.align_cameras_from_people`) put through the already validated
`pose.movement_metrics` chain (outlier rejection, smoothing, stillness
deadzone). Using the torso instead of the hip midpoint matters with ViTPose:
hips hidden by the table have no depth there any more, the shoulders still do.

Per camera for now: each camera's distance counts only what that camera
sees with depth. The two-camera fusion comes after the B->A alignment.

Usage:
    cd src
    python -m pose.session_overlay --session-dir ".../Sessions/10_individual_12" --camera camera_a
    # quick look at 2 minutes only:
    python -m pose.session_overlay --session-dir ... --camera camera_b --start-s 60 --end-s 180
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose.align_cameras_from_people import torso_roots
from pose.dwell_heatmap_3d import movement_from_roots
from pose.keypoints import COCO17, SKELETON_EDGES

ROLE_BGR = {"adult": (40, 40, 230), "child": (230, 120, 30), "uncertain": (160, 160, 160)}
ROLE_IT = {"adult": "adult", "child": "child", "uncertain": "?"}
MASK_THRESHOLD = 127


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def largest_blob_box(m: np.ndarray):
    n, labels, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[i, :4]
    return int(x), int(y), int(x + w), int(y + h)


def load_roles(arg: str | None, pose_dir: Path) -> dict[int, str]:
    if arg:
        return {int(k): v.strip() for k, v in (p.split(":") for p in arg.split(","))}
    f = pose_dir / "age_group.csv"
    if f.exists():
        t = pd.read_csv(f)
        return {int(r.global_person_id): r.role for r in t.itertuples()}
    print("[overlay] no roles given and no age_group.csv: everyone shown as uncertain")
    return {}


def floor_plane(mkv: Path, cache: Path):
    from pose.room_coverage import _valid_points, fit_floor_plane
    if not cache.exists():
        from pose.table_calibration import extract_reference
        extract_reference(mkv_path=str(mkv), out_prefix=str(cache).replace("_pointcloud.npy", ""), t_seconds=30)
    pc = _valid_points(np.load(cache))
    plane = fit_floor_plane(pc)
    print(f"[overlay] floor plane: rmse {plane['rmse_mm']:.1f} mm, {plane['n_inliers']} inliers")
    return plane, pc


class FloorMap:
    """Top-down map in the floor plane: grey furniture, per-role occupancy."""

    def __init__(self, plane, room_pc, size_px: int = 360, cell_mm: float = 100.0):
        # axes on the floor: u = the camera's left-right, v = away from the
        # camera, so the map reads like the video (camera at the bottom edge)
        n = np.asarray(plane["normal"], float)
        o = np.asarray(plane["point"], float)
        ex = np.array([1.0, 0.0, 0.0]) - n * n[0]
        ex /= np.linalg.norm(ex)
        ez = np.array([0.0, 0.0, 1.0]) - n * n[2]
        ez -= ex * (ez @ ex)
        ez /= np.linalg.norm(ez)
        self.fc = lambda p: np.c_[(np.asarray(p, float) - o) @ ex, (np.asarray(p, float) - o) @ ez,
                                  (np.asarray(p, float) - o) @ n]
        rng = np.random.default_rng(0)
        pts = room_pc[rng.choice(len(room_pc), min(60000, len(room_pc)), replace=False)]
        uvh = self.fc(pts)
        uvh = uvh[(uvh[:, 2] > -100) & (uvh[:, 2] < 2200)]
        lo, hi = np.percentile(uvh[:, :2], 1, 0) - 300, np.percentile(uvh[:, :2], 99, 0) + 300
        self.lo, self.cell = lo, cell_mm
        self.nu, self.nv = [int(np.ceil(x)) for x in (hi - lo) / cell_mm]
        self.scale = size_px / max(self.nu, self.nv)
        self.W, self.H = int(self.nu * self.scale), int(self.nv * self.scale)
        self.bg = np.full((self.H, self.W, 3), 245, np.uint8)
        furn = uvh[(uvh[:, 2] > 300) & (uvh[:, 2] < 1100)]
        floor = uvh[uvh[:, 2] <= 300]
        for arr, col in ((floor, (225, 225, 225)), (furn, (150, 150, 150))):
            px = self._px(arr[:, :2])
            ok = (px[:, 0] >= 0) & (px[:, 0] < self.W) & (px[:, 1] >= 0) & (px[:, 1] < self.H)
            self.bg[px[ok, 1], px[ok, 0]] = col
        self.occ = {"child": np.zeros((self.nv, self.nu)), "adult": np.zeros((self.nv, self.nu))}

    def _px(self, uv):
        p = (np.asarray(uv) - self.lo) / self.cell * self.scale
        return np.c_[p[:, 0], self.H - 1 - p[:, 1]].astype(int)

    def add(self, uv, role, weight):
        key = "child" if role == "child" else "adult"
        iu, iv = int((uv[0] - self.lo[0]) // self.cell), int((uv[1] - self.lo[1]) // self.cell)
        if 0 <= iu < self.nu and 0 <= iv < self.nv:
            self.occ[key][iv, iu] += weight

    def render(self, current: list[tuple[np.ndarray, str, int]]):
        img = self.bg.copy().astype(np.float32)
        for key, col in (("adult", np.array(ROLE_BGR["adult"])), ("child", np.array(ROLE_BGR["child"]))):
            o = cv2.GaussianBlur(self.occ[key], (0, 0), 1.0)
            if o.max() <= 0:
                continue
            a = np.log1p(o) / np.log1p(o.max())
            a = cv2.resize(np.flipud(a), (self.W, self.H), interpolation=cv2.INTER_LINEAR)[..., None] * 0.85
            img = img * (1 - a) + col * a
        img = img.astype(np.uint8)
        for uv, role, pid in current:
            p = self._px(np.asarray(uv)[None])[0]
            cv2.circle(img, tuple(p), 6, ROLE_BGR.get(role, ROLE_BGR["uncertain"]), -1)
            cv2.circle(img, tuple(p), 6, (255, 255, 255), 1)
            cv2.putText(img, str(pid), (p[0] + 7, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        cv2.rectangle(img, (0, 0), (self.W - 1, self.H - 1), (60, 60, 60), 1)
        cv2.putText(img, "top view", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
        return img


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Boxes by role, speed, distance, live floor map -- see docstring.")
    p.add_argument("--session-dir", required=True)
    p.add_argument("--camera", required=True, help="camera_a | camera_b")
    p.add_argument("--pose-subdir", default=None, help="default: <camera>_vitpose")
    p.add_argument("--roles", default=None, help="e.g. 0:adult,1:adult,2:child,3:adult (default: age_group.csv)")
    p.add_argument("--fps", type=float, default=None, help="default: from the video")
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--end-s", type=float, default=None)
    p.add_argument("--map-px", type=int, default=360)
    p.add_argument("--conf", type=float, default=0.3, help="keypoint confidence to draw")
    p.add_argument("--out", default=None, help="default: <pose dir>/overlay_<camera>.mp4")
    a = p.parse_args()

    S = Path(a.session_dir)
    cam = a.camera
    pose_dir = S / "pose" / (a.pose_subdir or f"{cam}_vitpose")
    video = S / "processed" / f"{cam}.mp4"
    mask_dir = S / "merged" / cam
    out = Path(a.out) if a.out else pose_dir / f"overlay_{cam}.mp4"

    cap = cv2.VideoCapture(str(video))
    fps = a.fps or cap.get(cv2.CAP_PROP_FPS) or 15.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(a.start_s * fps)
    f1 = min(n_total, int(a.end_s * fps)) if a.end_s else n_total

    roles = load_roles(a.roles, pose_dir)

    # --- 3D movement: robust torso -> validated movement chain
    df3d = pd.read_csv(pose_dir / "keypoints_3d.csv")
    roots = torso_roots(df3d)
    per_frame = movement_from_roots(roots, fps)
    per_frame = per_frame.sort_values(["global_person_id", "frame"])
    per_frame["disp"] = per_frame["displacement_mm"].fillna(0.0)
    per_frame["cum_m"] = per_frame.groupby("global_person_id")["disp"].cumsum() / 1000.0
    # display speed: short centred mean so the number is readable, not the raw per-frame value
    per_frame["speed_ms"] = (per_frame.groupby("global_person_id")["velocity_mm_s"]
                             .transform(lambda s: s.rolling(int(fps * 0.6) or 1, center=True, min_periods=1).mean())
                             / 1000.0)
    mv = per_frame.set_index(["frame", "global_person_id"])
    # last known total distance per person for every frame (frames without 3D keep the previous total)
    cum_ff = {}
    for pid, gpf in per_frame.groupby("global_person_id"):
        s_ = gpf.set_index("frame")["cum_m"].reindex(range(n_total)).ffill().fillna(0.0)
        cum_ff[int(pid)] = s_.to_numpy()

    # --- floor + live map
    plane, room_pc = floor_plane(S / f"{cam}.mkv", S / "calib" / f"{cam.replace('camera_', 'cam_')}_pointcloud.npy")
    fmap = FloorMap(plane, room_pc, size_px=a.map_px)
    xyz = per_frame[["X_mm", "Y_mm", "Z_mm"]].to_numpy()
    ok = np.isfinite(xyz).all(1)
    uv = np.full((len(per_frame), 2), np.nan)
    uv[ok] = fmap.fc(xyz[ok])[:, :2]
    per_frame["floor_u_mm"], per_frame["floor_v_mm"] = uv[:, 0], uv[:, 1]
    per_frame["role"] = per_frame["global_person_id"].map(lambda i: roles.get(int(i), "uncertain"))
    per_frame[["global_person_id", "frame", "role", "floor_u_mm", "floor_v_mm", "speed_ms", "cum_m"]].to_csv(
        pose_dir / f"movement_{cam}.csv", index=False)
    uv_idx = per_frame.set_index(["frame", "global_person_id"])[["floor_u_mm", "floor_v_mm"]]

    # --- posture (standing / seated / on the floor) from 3D shoulder height, + ELAN export
    from pose.posture_3d import LABEL_IT, classify_posture, elan_episodes
    post = classify_posture(df3d, plane, roles, fps=fps)
    post.to_csv(pose_dir / f"posture_{cam}.csv", index=False)
    elan_episodes(post, fps).to_csv(pose_dir / f"posture_{cam}_elan.tsv", sep="\t", index=False)
    posture_at = {(int(r.frame), int(r.global_person_id)): LABEL_IT.get(r.posture, r.posture)
                  for r in post.itertuples()}

    # occupancy before the window start (so a partial render still shows the history)
    hist = per_frame[(per_frame["frame"] < f0) & per_frame["floor_u_mm"].notna()]
    for r in hist.itertuples():
        fmap.add((r.floor_u_mm, r.floor_v_mm), r.role, 1.0 / fps)

    # --- 2D skeletons
    kp = pd.read_csv(pose_dir / "keypoints_smoothed.csv")
    xcol, ycol = ("x_smooth", "y_smooth") if "x_smooth" in kp.columns else ("x", "y")
    kp = kp.dropna(subset=[xcol, ycol])
    kp = kp[kp["confidence"].fillna(1.0) >= a.conf]
    if "in_own_mask" in kp.columns:
        kp = kp[kp["in_own_mask"].astype(str) != "False"]
    kp_by_frame = {f: g for f, g in kp[(kp["frame"] >= f0) & (kp["frame"] < f1)].groupby("frame")}
    edges = [(COCO17.index(x), COCO17.index(y)) for x, y in SKELETON_EDGES]

    mcaps = {int(f.stem): cv2.VideoCapture(str(f)) for f in sorted(mask_dir.glob("*.mp4")) if f.stem.isdigit()}
    for c in [cap, *mcaps.values()]:
        c.set(cv2.CAP_PROP_POS_FRAMES, f0)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    print(f"[overlay] {cam}: frames {f0}-{f1}, {len(mcaps)} ids, roles {roles} -> {out}")

    for f in range(f0, f1):
        ok, img = cap.read()
        if not ok:
            break
        current = []
        g = kp_by_frame.get(f)
        for pid, mc in mcaps.items():
            okm, m = mc.read()
            if not okm:
                continue
            mask = (m[:, :, 0] if m.ndim == 3 else m) > MASK_THRESHOLD
            if not mask.any():
                continue
            role = roles.get(pid, "uncertain")
            col = ROLE_BGR.get(role, ROLE_BGR["uncertain"])
            box = largest_blob_box(mask)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            # skeleton
            if g is not None:
                gp = g[g["global_person_id"] == pid]
                pts = {COCO17.index(r.keypoint_name): (int(getattr(r, xcol)), int(getattr(r, ycol)))
                       for r in gp.itertuples() if r.keypoint_name in COCO17}
                for i, j in edges:
                    if i in pts and j in pts:
                        cv2.line(img, pts[i], pts[j], (255, 255, 255), 3)
                        cv2.line(img, pts[i], pts[j], col, 1)
            # label: id, role, speed, distance
            key = (f, pid)
            speed = mv["speed_ms"].get(key, np.nan) if key in mv.index else np.nan
            dist = mv["cum_m"].get(key, np.nan) if key in mv.index else np.nan
            if not np.isfinite(dist):  # no 3D this frame: last known total
                dist = cum_ff[pid][f] if pid in cum_ff and f < len(cum_ff[pid]) else 0.0
            lines = [f"id {pid}  {ROLE_IT.get(role, role)}  {posture_at.get((f, pid), '')}".rstrip(),
                     f"{speed:.2f} m/s" if np.isfinite(speed) else "-- m/s",
                     f"{dist:.1f} m"]
            ty = max(0, y1 - 8 - 22 * (len(lines) - 1))
            for k, line in enumerate(lines):
                yy = ty + 22 * k
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, yy - th - 4), (x1 + tw + 6, yy + 4), col, -1)
                cv2.putText(img, line, (x1 + 3, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # live map
            if key in uv_idx.index:
                u, v = uv_idx.loc[key]
                if np.isfinite(u):
                    fmap.add((u, v), role, 1.0 / fps)
                    current.append(((u, v), role, pid))
        inset = fmap.render(current)
        h_, w_ = inset.shape[:2]
        y0, x0 = H - h_ - 10, W - w_ - 10
        img[y0:y0 + h_, x0:x0 + w_] = inset
        cv2.putText(img, f"t={f / fps:7.1f}s", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(img)
        if (f - f0) % 1500 == 0:
            print(f"[overlay] frame {f}/{f1}")
    writer.release()

    final = fmap.render([])
    cv2.imwrite(str(pose_dir / f"occupancy_{cam}.png"), cv2.resize(final, None, fx=2, fy=2,
                                                                    interpolation=cv2.INTER_NEAREST))
    summ = per_frame.groupby(["global_person_id", "role"]).agg(
        distance_m=("cum_m", "max"), mean_speed_ms=("speed_ms", "mean"),
        frames_with_3d=("X_mm", lambda s: int(s.notna().sum())))
    print(summ.round(2).to_string())
    share = (post.groupby(["global_person_id", "role"])["posture"].value_counts(normalize=True)
             .unstack(fill_value=0).rename(columns=LABEL_IT) * 100)
    print("\n[overlay] posture, % of frames with 3D shoulders:")
    print(share.round(1).to_string())
    print(f"[overlay] wrote {out}, occupancy_{cam}.png, movement_{cam}.csv, posture_{cam}.csv, "
          f"posture_{cam}_elan.tsv")


if __name__ == "__main__":
    main()
