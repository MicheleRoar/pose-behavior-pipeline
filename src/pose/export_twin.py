"""
pose/export_twin.py
=====================
"Digital twin" of a session: ONE self-contained HTML page (open it locally
in a browser, nothing is uploaded) showing the room as a 3D point cloud from
the Kinect depth, with the people's 3D skeletons moving inside it over time.

  - skeletons coloured by role (red adult, blue child, grey uncertain),
    a short floor trail per person, label with id / role / posture / speed;
  - orbit with the mouse, presets (top view, camera A, camera B), play /
    pause, timeline, playback speed;
  - the room is a PEOPLE-FREE background: depth and colour sampled at ~40
    instants over the recording, per pixel the "farthest" reading kept
    (people are always in front of the walls), so nobody -- and no face --
    is frozen into the room. Real colours by default, `--height-colors` for
    a neutral height colouring.

Camera B is merged in when the alignment from `pose.align_cameras_from_people`
is given: its point cloud and its people are moved into camera A's frame;
a person matched in both cameras gets, per joint, the mean of the two views
where both have it (otherwise whichever has it), so a joint hidden from one
camera is still drawn from the other.

Joint depths are first corrected with the skeleton itself
(`pose.lift_skeleton`): a joint whose single depth pixel hit the wall or the
table is put back on its own viewing ray at the person's real bone length,
and hidden joints get a position the same way. Then positions are lightly
smoothed in time (centred median over
`--smooth-frames`) before export: depth jitter on single joints is what
makes raw 3D skeletons look nervous.

World frame of the page: floor = y=0 (floor plane fitted on camera A's own
point cloud), y up, metres.

Usage:
    cd src
    python -m pose.export_twin --session-dir ".../Sessions/10_individual_12" \\
        [--alignment-json .../calib/camera_b_to_camera_a_people.json] [--start-s 60 --end-s 360]
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

from pose.keypoints import COCO17, SKELETON_EDGES

ROLE_HEX = {"adult": "#e8473a", "child": "#2f7fe0", "uncertain": "#9a9a9a"}


def _pc(session: Path, cam: str):
    """Room point cloud (+ colour image) of one camera, extracted once from the mkv."""
    from pose.table_calibration import extract_reference
    prefix = session / "calib" / cam.replace("camera_", "cam_")
    pc_path, col_path = Path(f"{prefix}_pointcloud.npy"), Path(f"{prefix}_color.png")
    if not pc_path.exists():
        extract_reference(mkv_path=str(session / f"{cam}.mkv"), out_prefix=str(prefix), t_seconds=30)
    pc = np.load(pc_path)
    col = cv2.imread(str(col_path)) if col_path.exists() else None
    return pc, col


def _room_background(session: Path, cam: str, n_samples: int = 40, keep_q: float = 0.75):
    """People-free room: depth sampled at `n_samples` instants spread over the
    whole recording; per pixel the sample at the `keep_q` quantile of depth is
    kept. People are always IN FRONT of the background, so they only ever make
    a pixel closer: a high quantile keeps the wall / floor / furniture behind
    them. Colour from the same sample, so the room has no people (and no
    faces) in it. Cached in calib/<cam>_room_bg.npz."""
    cache = session / "calib" / f"{cam.replace('camera_', 'cam_')}_room_bg.npz"
    if cache.exists():
        z = np.load(cache)
        return z["pc"], z["rgb"]
    from pyk4a import PyK4APlayback, SeekOrigin
    pb = PyK4APlayback(str(session / f"{cam}.mkv"))
    pb.open()
    length = pb.length  # usec
    pcs, cols = [], []
    try:
        for k in range(n_samples):
            pb.seek(int((k + 0.5) / n_samples * length), SeekOrigin.BEGIN)
            for _ in range(30):
                try:
                    c = pb.get_next_capture()
                except EOFError:
                    break
                if c.transformed_depth_point_cloud is None or c.color is None:
                    continue
                col = cv2.imdecode(np.frombuffer(c.color, np.uint8), cv2.IMREAD_COLOR)
                if col is None:
                    continue
                pcs.append(c.transformed_depth_point_cloud.astype(np.int16))
                cols.append(col)
                break
    finally:
        pb.close()
    P = np.stack(pcs)                                  # (n, H, W, 3)
    Z = P[..., 2].astype(np.float32)
    Z[Z <= 0] = np.nan
    target = np.nanquantile(Z, keep_q, axis=0)         # (H, W)
    diff = np.abs(Z - target[None])
    diff[~np.isfinite(diff)] = np.inf
    pick = np.argmin(diff, axis=0)
    ii, jj = np.indices(target.shape)
    pc = P[pick, ii, jj]
    pc[~np.isfinite(target)] = 0
    rgb = np.stack(cols)[pick, ii, jj][..., ::-1]      # BGR -> RGB
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, pc=pc, rgb=rgb)
    print(f"[twin] people-free room for {cam} from {len(pcs)} instants -> {cache}")
    return pc, rgb


def _cloud_points(pc, col, n_max, rng, edge_frac: float = 0.03):
    """Valid points of an (H, W, 3) cloud, minus 'flying pixels': on depth
    edges the sensor returns points floating between foreground and
    background; a pixel whose depth differs from a neighbour's by more than
    `edge_frac` of its own depth is dropped."""
    H, W = pc.shape[:2]
    Z = pc[..., 2].astype(np.float32)
    jump = np.zeros_like(Z)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        sh = np.roll(np.roll(Z, dy, 0), dx, 1)
        jump = np.maximum(jump, np.where(sh > 0, np.abs(Z - sh), 0))
    P = pc.reshape(-1, 3).astype(np.float64)
    ok = (P[:, 2] > 0) & (jump.reshape(-1) <= edge_frac * np.maximum(Z.reshape(-1), 1))
    rgb = None
    if col is not None and col.shape[:2] == (H, W):
        rgb = col.reshape(-1, 3)[ok]  # already RGB
    P = P[ok]
    idx = rng.choice(len(P), min(n_max, len(P)), replace=False)
    return P[idx], (rgb[idx] if rgb is not None else None)


def _world_frame(plane):
    """Rotation/translation camera_a(mm) -> world (m): floor y=0, y up,
    x = camera A's left-right, -z = away from camera A."""
    n = np.asarray(plane["normal"], float)
    o = np.asarray(plane["point"], float)
    ex = np.array([1.0, 0.0, 0.0]) - n * n[0]
    ex /= np.linalg.norm(ex)
    ez = np.cross(ex, n)  # completes a right-handed frame (x, y=up, z)
    R = np.stack([ex, n, ez])
    return lambda P: ((np.asarray(P, float) - o) @ R.T) / 1000.0


def _measured_mask(df3d: pd.DataFrame, frames: np.ndarray) -> dict[int, np.ndarray]:
    """{pid: (n_frames, 17) bool} -- joint has a real (not lifted) depth reading."""
    out = {}
    fidx = {f: i for i, f in enumerate(frames)}
    d = df3d.dropna(subset=["Z_mm"])
    if "lifted" in d.columns:
        d = d[~d["lifted"].astype(bool)]
    for pid, g in d.groupby("global_person_id"):
        M = np.zeros((len(frames), 17), bool)
        g = g[g["frame"].isin(fidx) & g["keypoint_name"].isin(COCO17)]
        M[g["frame"].map(fidx).to_numpy(), g["keypoint_name"].map(COCO17.index).to_numpy()] = True
        out[int(pid)] = M
    return out


def _skeleton_tensor(df3d: pd.DataFrame, frames: np.ndarray, smooth: int) -> dict[int, np.ndarray]:
    """{pid: (n_frames, 17, 3) mm, NaN where missing}, median-smoothed in time."""
    out = {}
    fidx = {f: i for i, f in enumerate(frames)}
    for pid, g in df3d.dropna(subset=["X_mm", "Y_mm", "Z_mm"]).groupby("global_person_id"):
        T = np.full((len(frames), 17, 3), np.nan)
        g = g[g["frame"].isin(fidx) & g["keypoint_name"].isin(COCO17)]
        fi = g["frame"].map(fidx).to_numpy()
        ki = g["keypoint_name"].map(COCO17.index).to_numpy()
        T[fi, ki] = g[["X_mm", "Y_mm", "Z_mm"]].to_numpy()
        if smooth > 1:
            flat = pd.DataFrame(T.reshape(len(frames), -1))
            T = flat.rolling(smooth, center=True, min_periods=1).median().to_numpy().reshape(T.shape)
            # rolling median fills gaps shorter than the window from neighbours; keep long gaps empty
        out[int(pid)] = clean_skeleton(T)
    return out


def clean_room(P: np.ndarray, rgb: np.ndarray | None, *, fine: float = 0.015, coarse: float = 0.06,
               min_coarse: int = 6, y_max: float = 2.3):
    """World-frame room cloud (m) -> tidy cloud: floor-to-`y_max` slab only
    (no ceiling and lights), inside the room's footprint, isolated specks
    removed (coarse voxels with fewer than `min_coarse` points), then evened
    out to one point per `fine` voxel (mean colour), so the two cameras'
    overlap does not look denser/noisier than the rest."""
    keep = (P[:, 1] > -0.05) & (P[:, 1] < y_max)
    lo, hi = np.percentile(P[keep][:, [0, 2]], [0.5, 99.5], axis=0)
    keep &= (P[:, 0] > lo[0] - 0.2) & (P[:, 0] < hi[0] + 0.2) & (P[:, 2] > lo[1] - 0.2) & (P[:, 2] < hi[1] + 0.2)
    P = P[keep]
    rgb = rgb[keep] if rgb is not None else None
    kc = np.floor(P / coarse).astype(np.int64)
    _, inv, cnt = np.unique(kc, axis=0, return_inverse=True, return_counts=True)
    dense = cnt[inv.reshape(-1)] >= min_coarse
    P = P[dense]
    rgb = rgb[dense] if rgb is not None else None
    kf = np.floor(P / fine).astype(np.int64)
    _, idx, inv = np.unique(kf, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    n = len(idx)
    Pm = np.zeros((n, 3))
    np.add.at(Pm, inv, P)
    c = np.bincount(inv, minlength=n)[:, None]
    Pm /= c
    if rgb is not None:
        Cm = np.zeros((n, 3))
        np.add.at(Cm, inv, rgb.astype(np.float64))
        rgb = (Cm / c).astype(np.uint8)
    return Pm.astype(np.float32), rgb


TORSO = [COCO17.index(k) for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
MAX_FROM_TORSO_MM = 1100.0   # no joint is further than this from the torso centre
MAX_BONE_MM = 750.0          # no limb segment is longer than this


def clean_skeleton(T: np.ndarray) -> np.ndarray:
    """Drop joints whose depth reading clearly belongs to something else (a
    wall behind, the table in front): too far from the person's torso, or
    making a bone longer than any real bone. They would draw as long lines
    across the room."""
    T = T.copy()
    torso = np.nanmedian(T[:, TORSO], axis=1)                     # (n, 3)
    d = np.linalg.norm(T - torso[:, None], axis=2)
    T[d > MAX_FROM_TORSO_MM] = np.nan
    for _ in range(2):
        for a, b in SKELETON_EDGES:
            ia, ib = COCO17.index(a), COCO17.index(b)
            L = np.linalg.norm(T[:, ia] - T[:, ib], axis=1)
            bad = L > MAX_BONE_MM
            far_b = np.linalg.norm(T[:, ib] - torso, axis=1) >= np.linalg.norm(T[:, ia] - torso, axis=1)
            T[bad & far_b, ib] = np.nan
            T[bad & ~far_b, ia] = np.nan
    return T


def fuse_views(TA: np.ndarray, TB: np.ndarray, agree_mm: float = 300.0, *, primary: bool = True,
               MA: np.ndarray | None = None, MB: np.ndarray | None = None) -> np.ndarray:
    """primary=True (default): per frame, the whole skeleton comes from the
    camera that MEASURED more of this person's joints (MA/MB: boolean
    'measured, not lifted' masks), the other camera only fills joints the
    first lacks -- no blending of two slightly different views of the same
    body (one sees the front, the other the back), which is what makes
    averaged skeletons look mushy. primary=False: per-joint mean when the two
    agree, otherwise the one closer to the fused torso."""
    if primary and MA is not None and MB is not None:
        # which camera sees more, decided over ~2 s windows (not per frame), so
        # the skeleton does not flip between the two views every few frames
        score = pd.Series((MA.sum(1) - MB.sum(1)).astype(float))
        use_a = (score.rolling(31, center=True, min_periods=1).mean() >= 0).to_numpy()
        first = np.where(use_a[:, None, None], TA, TB)
        second = np.where(use_a[:, None, None], TB, TA)
        return np.where(np.isnan(first), second, first)
    both = ~np.isnan(TA[..., 0]) & ~np.isnan(TB[..., 0])
    out = np.where(np.isnan(TA), TB, TA)
    torso = np.nanmean(np.stack([np.nanmedian(TA[:, TORSO], 1), np.nanmedian(TB[:, TORSO], 1)]), 0)
    gap = np.linalg.norm(TA - TB, axis=2)
    mean = (TA + TB) / 2
    dA = np.linalg.norm(TA - torso[:, None], axis=2)
    dB = np.linalg.norm(TB - torso[:, None], axis=2)
    pick = np.where((dA <= dB)[..., None], TA, TB)
    agree = both & (gap <= agree_mm)
    disagree = both & (gap > agree_mm)
    out[agree] = mean[agree]
    out[disagree] = pick[disagree]
    return out


def _torso_mm(T):
    return np.nanmedian(T[:, TORSO], axis=1)


def fused_torso_mm(views: list[np.ndarray], disagree_mm: float = 500.0) -> np.ndarray:
    """(n, 3) torso centre (mm, camera A frame) from one or two camera views
    of the same person; camera B is first shifted by the person's median
    A-B offset (front vs back of the body), so the two can be averaged and
    switching between them does not make the person jump."""
    tor = [_torso_mm(T) for T in views]
    if len(tor) == 1:
        return tor[0]
    both = np.isfinite(tor[0][:, 0]) & np.isfinite(tor[1][:, 0])
    bias = np.nanmedian(tor[0][both] - tor[1][both], axis=0) if both.sum() > 30 else np.zeros(3)
    A, B = tor[0], tor[1] + bias
    out = np.nanmean(np.stack([A, B]), axis=0)
    # where the two cameras disagree by more than `disagree_mm` one of them is
    # tracking something else for a moment (a wrong id, a depth error): do
    # not average two different places, keep the one continuing the track
    gap = np.linalg.norm(A - B, axis=1)
    last = None
    for i in range(len(out)):
        if np.isfinite(gap[i]) and gap[i] > disagree_mm and last is not None:
            out[i] = A[i] if np.linalg.norm(A[i] - last) <= np.linalg.norm(B[i] - last) else B[i]
        if np.isfinite(out[i, 0]):
            last = out[i]
    return out


def shoulder_mid_mm(views: list[np.ndarray]) -> np.ndarray:
    sh_idx = [COCO17.index("left_shoulder"), COCO17.index("right_shoulder")]
    return np.nanmean(np.stack([np.nanmean(T[:, sh_idx], axis=1) for T in views]), axis=0)


def person_anchor(views: list[np.ndarray], to_w, fps: float, *, window_s: float = 1.0,
                  hold_m: float = 0.10) -> np.ndarray:
    """A steady (n, 3) world position per person for the avatar / top view:
    x, z = torso centre on the floor, y = shoulder height.
    - two cameras: camera B's torso is shifted by the person's median A-B
      offset (one camera sees the front of the body, the other the back:
      a constant ~10-30 cm gap) and then averaged with A's, so switching
      between cameras does not make the person jump;
    - centred moving average over `window_s`;
    - hold: the floor position only moves once it has drifted more than
      `hold_m` from where the person was last placed -- a seated child's
      residual jitter stays on the spot, real movement goes through."""
    tor = fused_torso_mm(views)
    sh = shoulder_mid_mm(views)
    Wt, Ws = to_w(tor), to_w(sh)
    w = max(1, int(window_s * fps))
    df = pd.DataFrame({"x": Wt[:, 0], "z": Wt[:, 2], "y": Ws[:, 1]})
    sm = df.rolling(w, center=True, min_periods=max(1, w // 3)).mean().to_numpy()
    out = np.full((len(sm), 3), np.nan)
    held = None
    for i, (x, z, y) in enumerate(sm):
        if not np.isfinite(x):
            continue
        if held is None or np.hypot(x - held[0], z - held[1]) > hold_m:
            held = (x, z)
        out[i] = (held[0], y, held[1])
    return out


def load_session_people(S: Path, *, alignment: dict | None, fps: float = 15.0, start_s: float = 0.0,
                        end_s: float | None = None, smooth_frames: int = 9, pose_suffix: str = "_vitpose",
                        lift: bool = True, extra_pairs: str | None = None, room_samples: int = 40,
                        log: str = "twin") -> dict:
    """Everything about the people of a session in ONE world frame, shared by
    the digital twin and `pose.session_metrics`:
      plane / to_w     camera A's floor plane and camera_a(mm) -> world(m)
      frames           frame indices (camera A's clock)
      sk               {pid: (n, 17, 3) mm, camera A frame} fused skeletons
      views            {pid: [skeleton per camera]} (for steady positions)
      labels, roles    'A2/B2', 'child' ...
      bg               {camera: people-free room point cloud, rgb}
    pid = camera A id, or 1000 + camera B id for people seen only by B."""
    from pose.room_coverage import _valid_points, fit_floor_plane

    pc_a, _ = _pc(S, "camera_a")
    plane = fit_floor_plane(_valid_points(pc_a))
    to_w = _world_frame(plane)
    print(f"[{log}] floor (camera A): rmse {plane['rmse_mm']:.1f} mm")
    al = alignment
    R_ba = np.asarray(al["rotation_3x3"]) if al else None
    t_ba = np.asarray(al["translation_mm"]) if al else None
    off = int(al.get("frame_offset_b_to_a", 0)) if al else 0
    bg = {"camera_a": _room_background(S, "camera_a", room_samples)}
    if al:
        bg["camera_b"] = _room_background(S, "camera_b", room_samples)

    def pose_dir(cam):
        return S / "pose" / f"{cam}{pose_suffix}"

    def table(cam):
        f = pose_dir(cam) / "age_group.csv"
        return pd.read_csv(f) if f.exists() else None

    def load_3d(cam):
        df = pd.read_csv(pose_dir(cam) / "keypoints_3d.csv", low_memory=False)
        if not lift:
            return df
        from pose.lift_skeleton import fit_intrinsics, lift_dataframe
        K = fit_intrinsics(bg[cam][0])
        if not (200 < K[0] < 5000 and 200 < K[1] < 5000):
            print(f"[{log}] {cam}: implausible intrinsics {K} -- skeleton lifting skipped")
            return df
        t = table(cam)
        sizes = {int(r.global_person_id): float(r.size_index) for r in t.itertuples()
                 if np.isfinite(r.size_index)} if t is not None else {}
        out = lift_dataframe(df, K, scale_hint=sizes)
        print(f"[{log}] {cam}: intrinsics fx={K[0]:.0f} fy={K[1]:.0f} cx={K[2]:.0f} cy={K[3]:.0f}; "
              f"{out['lifted'].mean():.1%} of joints re-placed on their ray at the bone length")
        return out

    def roles_of(cam):
        t = table(cam)
        return {int(r.global_person_id): r.role for r in t.itertuples()} if t is not None else {}

    df_a = load_3d("camera_a")
    n_total = int(df_a["frame"].max()) + 1
    f0 = int(start_s * fps)
    f1 = min(n_total, int(end_s * fps)) if end_s else n_total
    frames = np.arange(f0, f1)
    sk = _skeleton_tensor(df_a, frames, smooth_frames)
    meas_a = _measured_mask(df_a, frames)
    roles = roles_of("camera_a")
    labels = {pid: f"A{pid}" for pid in sk}
    views = {pid: [T] for pid, T in sk.items()}
    if al:
        df_b = load_3d("camera_b").copy()
        df_b["frame"] = df_b["frame"] + off
        df_b[["X_mm", "Y_mm", "Z_mm"]] = df_b[["X_mm", "Y_mm", "Z_mm"]].to_numpy() @ R_ba.T + t_ba
        sk_b = _skeleton_tensor(df_b, frames, smooth_frames)
        meas_b = _measured_mask(df_b, frames)
        roles_b = roles_of("camera_b")
        b_to_a = {int(d["id_b"]): int(d["id_a"]) for d in al["assignment"]}
        if extra_pairs:
            for pr in extra_pairs.split(","):
                ia, ib = (int(x) for x in pr.split(":"))
                b_to_a[ib] = ia
        for pb, T in sk_b.items():
            if pb in b_to_a and b_to_a[pb] in sk:
                pa = b_to_a[pb]
                sk[pa] = fuse_views(sk[pa], T, MA=meas_a.get(pa), MB=meas_b.get(pb))
                views[pa].append(T)
                labels[pa] = f"A{pa}/B{pb}"
            else:
                key = 1000 + pb
                sk[key] = T
                views[key] = [T]
                roles[key] = roles_b.get(pb, "uncertain")
                labels[key] = f"B{pb}"
    cams = {"A": to_w(np.zeros((1, 3)))[0].tolist()}
    if al:
        cams["B"] = to_w((np.zeros((1, 3)) @ R_ba.T + t_ba))[0].tolist()
    return {"plane": plane, "to_w": to_w, "frames": frames, "sk": sk, "views": views, "labels": labels,
            "roles": roles, "bg": bg, "R_ba": R_ba, "t_ba": t_ba, "cams": cams}


def main() -> None:
    p = argparse.ArgumentParser(description="3D digital twin of a session as a local HTML page -- see docstring.")
    p.add_argument("--session-dir", required=True)
    p.add_argument("--pose-suffix", default="_vitpose", help="pose/<camera><suffix>/")
    p.add_argument("--alignment-json", default=None, help="camera_b -> camera_a (align_cameras_from_people)")
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--end-s", type=float, default=None)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--smooth-frames", type=int, default=9)
    p.add_argument("--no-lift", action="store_true",
                   help="draw the raw per-pixel depth joints (no bone-length correction, see pose/lift_skeleton.py)")
    p.add_argument("--cloud-points", type=int, default=250000, help="per camera")
    p.add_argument("--room-samples", type=int, default=40, help="instants used for the people-free room")
    p.add_argument("--height-colors", action="store_true",
                   help="colour the room by height instead of its real colours")
    p.add_argument("--extra-pairs", default=None,
                   help="A:B id pairs to merge on top of the alignment's assignment, e.g. 3:3 (people the "
                        "alignment left unmatched because they co-occur too little)")
    p.add_argument("--metrics-dir", default=None,
                   help="pose.session_metrics output (default: <session>/metrics) -- posture, speed, distances")
    p.add_argument("--out", default=None, help="default: <session>/twin.html")
    a = p.parse_args()

    S = Path(a.session_dir)
    rng = np.random.default_rng(0)
    al = json.loads(Path(a.alignment_json).read_text()) if a.alignment_json else None
    ses = load_session_people(S, alignment=al, fps=a.fps, start_s=a.start_s, end_s=a.end_s,
                              smooth_frames=a.smooth_frames, pose_suffix=a.pose_suffix, lift=not a.no_lift,
                              extra_pairs=a.extra_pairs, room_samples=a.room_samples)
    to_w, frames, sk = ses["to_w"], ses["frames"], ses["sk"]

    # --- room (people-free background)
    clouds = []
    for cam, (bg, rgb) in ses["bg"].items():
        P, C = _cloud_points(bg, rgb, a.cloud_points, rng)
        if cam == "camera_b":
            P = P @ ses["R_ba"].T + ses["t_ba"]
        clouds.append((to_w(P), C))
    cloud = np.vstack([c[0] for c in clouds]).astype(np.float32)
    rgb_all = np.vstack([c[1] for c in clouds]) if all(c[1] is not None for c in clouds) else None
    cloud, rgb_all = clean_room(cloud, rgb_all)
    if not a.height_colors and rgb_all is not None:
        colors = rgb_all.astype(np.uint8)
    else:
        h = np.clip(cloud[:, 1] / 2.4, 0, 1)
        colors = (np.c_[0.55 + 0.35 * h, 0.58 + 0.30 * h, 0.62 + 0.30 * h] * 255).astype(np.uint8)

    # --- per-frame posture / speed / distance from pose.session_metrics (fused A+B), if present
    mdir = Path(a.metrics_dir) if a.metrics_dir else S / "metrics"
    met = None
    if (mdir / "metrics_per_frame.csv").exists():
        met = pd.read_csv(mdir / "metrics_per_frame.csv").set_index(["pid", "frame"])
        print(f"[twin] posture/speed/distance from {mdir / 'metrics_per_frame.csv'}")
    else:
        print(f"[twin] no {mdir / 'metrics_per_frame.csv'} -- run pose.session_metrics for posture/speed labels")

    people = []
    for pid, T in sk.items():
        W = to_w(T.reshape(-1, 3)).reshape(T.shape).astype(np.float32)
        post, speed, dist = [], [], []
        if met is not None:
            sub = met.xs(pid, level="pid") if pid in met.index.get_level_values(0) else None
            if sub is not None:
                sub = sub.reindex(frames)
                post = [x if isinstance(x, str) else "" for x in sub["posture"]]
                speed = [None if pd.isna(x) else round(float(x), 2) for x in sub["speed_ms"]]
                dist = [None if pd.isna(x) else round(float(x), 1) for x in sub["distance_m"]]
        anchor = person_anchor(ses["views"][pid], to_w, a.fps)
        role = ses["roles"].get(pid, "uncertain")
        people.append({
            "anchor": base64.b64encode(np.nan_to_num(anchor, nan=-999.0).astype(np.float32).tobytes()).decode(),
            "label": ses["labels"][pid], "role": role, "color": ROLE_HEX.get(role, "#9a9a9a"),
            "data": base64.b64encode(np.nan_to_num(W, nan=-999.0).tobytes()).decode(),
            "posture": post, "speed": speed, "dist": dist,
        })

    payload = {
        "fps": a.fps, "start_frame": int(frames[0]), "n_frames": int(len(frames)),
        "edges": [[COCO17.index(x), COCO17.index(y)] for x, y in SKELETON_EDGES],
        "cloud": base64.b64encode(cloud.tobytes()).decode(),
        "cloud_colors": base64.b64encode(colors.tobytes()).decode(),
        "people": people, "cams": ses["cams"],
        "title": f"{S.name} — digital twin" + (" (A+B)" if al else " (camera A)"),
    }
    tpl = (Path(__file__).with_name("twin_template.html")).read_text()
    html = tpl.replace("/*__PAYLOAD__*/null", json.dumps(payload))
    out = Path(a.out) if a.out else S / "twin.html"
    out.write_text(html)
    print(f"[twin] {len(people)} people, {len(frames)} frames, {len(cloud)} room points -> {out} "
          f"({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
