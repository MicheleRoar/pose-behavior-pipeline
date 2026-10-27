"""
pose/compare_posture_elan.py
==============================
Agreement between our 3D posture (`pose.posture_3d`, via
`pose.session_overlay`) and the clinicians' ELAN posture codes, read from the
annotation coverage report (sheet "Events", normalised keys):

  child      Type1.cX_CP.CST standing, CSI sitting, CLF on the floor lying,
             CCR crouching, CHO hovering/leaning over the table, CRE reaching,
             CGO gone
  therapist  Type1.tX_TP.TST, TSI, TLF, TCR, THO, TRE, TGO

Our three classes map as: standing = ST, seated = SI, floor = LF + CR.
HO (hovering/leaning) and RE (reaching) are in-between postures our 3-class
rule does not model yet: they are reported in the confusion table but left
out of the agreement percentage, as is GO (person gone).

Time alignment: ELAN times may be offset from the video (audio offset). The
offset is ESTIMATED by trying shifts in [-max, +max] s and keeping the one
with the best agreement for the child -- printed, so it can be checked
against the session's known `audio_time_offset`, or fixed with --offset-s.

The therapist tier does not say which adult it is when two therapists are
present: every adult id is compared and the best-matching one is reported.

Usage:
    cd src
    python -m pose.compare_posture_elan \\
        --posture-csv ".../pose/camera_a_vitpose/posture_camera_a.csv" \\
        --report ".../ADVANCE_annotation_coverage_report_20260911.xlsx" \\
        --session 12_10 --child-id 2 --fps 15
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

OURS = {"standing": "ST", "seated": "SI", "floor": "FL"}
ELAN_TO_CLASS = {"ST": "ST", "SI": "SI", "LF": "FL", "CR": "FL", "HO": "HO", "RE": "RE", "GO": "GO"}


def load_elan(report: str, session: str) -> pd.DataFrame:
    if report.endswith((".xlsx", ".xls")):
        ev = pd.read_excel(report, sheet_name="Events", header=5)
    else:
        ev = pd.read_csv(report)
    ev = ev[(ev["Session ID"].astype(str) == session) & (ev["Included"] == True)]  # noqa: E712
    key = ev["Normalised key"].astype(str)
    ev = ev[key.str.contains(r"\.cX_CP\.|\.tX_TP\.", regex=True)].copy()
    ev["who"] = np.where(ev["Normalised key"].str.contains(".cX_CP."), "child", "therapist")
    ev["code"] = ev["Normalised key"].str.split(".").str[-1].str[1:]  # CST -> ST, TSI -> SI
    ev["cls"] = ev["code"].map(ELAN_TO_CLASS)
    return ev[["who", "code", "cls", "Start (ms)", "End (ms)"]].dropna()


def elan_timeline(ev: pd.DataFrame, who: str, frames: np.ndarray, fps: float, offset_s: float) -> np.ndarray:
    """ELAN class per video frame (None where nothing is annotated). Later
    events overwrite earlier ones where they overlap."""
    lab = np.full(len(frames), None, dtype=object)
    t = frames / fps
    for r in ev[ev["who"] == who].sort_values("Start (ms)").itertuples():
        s, e = r._4 / 1000.0 - offset_s, r._5 / 1000.0 - offset_s
        lab[(t >= s) & (t < e)] = r.cls
    return lab


def agreement(ours: pd.Series, elan: np.ndarray) -> tuple[float, int]:
    o = ours.to_numpy()
    m = np.array([e in ("ST", "SI", "FL") for e in elan]) & pd.notna(o)
    if m.sum() == 0:
        return np.nan, 0
    return float((o[m] == elan[m]).mean()), int(m.sum())


def main() -> None:
    p = argparse.ArgumentParser(description="Our posture vs ELAN posture codes -- see module docstring.")
    p.add_argument("--posture-csv", required=True)
    p.add_argument("--report", required=True, help="annotation coverage report .xlsx (or its Events sheet as .csv)")
    p.add_argument("--session", required=True, help="Session ID as in the report, e.g. 12_10")
    p.add_argument("--child-id", type=int, required=True, help="our global_person_id of the child")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--offset-s", type=float, default=None, help="ELAN time - video time; estimated if omitted")
    p.add_argument("--max-offset-s", type=float, default=120.0)
    a = p.parse_args()

    post = pd.read_csv(a.posture_csv)
    if "global_person_id" not in post.columns and "pid" in post.columns:  # session_metrics output
        post = post.rename(columns={"pid": "global_person_id"})
    post["cls"] = post["posture"].map(OURS)
    ev = load_elan(a.report, a.session)
    if ev.empty:
        raise SystemExit(f"No posture events for session {a.session} in {a.report}")
    print(f"[elan] {len(ev)} posture events: " + ", ".join(f"{k}={v}" for k, v in ev.groupby(['who', 'code']).size().items()))

    child = post[post["global_person_id"] == a.child_id].set_index("frame")["cls"]
    frames = child.index.to_numpy()

    if a.offset_s is None:
        best = (-1, 0.0)
        for off in np.arange(-a.max_offset_s, a.max_offset_s + 0.01, 0.5):
            acc, n = agreement(child, elan_timeline(ev, "child", frames, a.fps, off))
            if n > a.fps * 30 and acc > best[0]:
                best = (acc, off)
        offset = best[1]
        print(f"[elan] estimated offset (ELAN - video): {offset:+.1f} s  (child agreement {best[0]:.1%} there)")
    else:
        offset = a.offset_s

    def report(name, ours_series, who):
        fr = ours_series.index.to_numpy()
        el = elan_timeline(ev, who, fr, a.fps, offset)
        acc, n = agreement(ours_series, el)
        ct = pd.crosstab(pd.Series(el, name="ELAN"), pd.Series(ours_series.to_numpy(), name="ours"))
        print(f"\n=== {name}: agreement {acc:.1%} on {n / a.fps / 60:.1f} min annotated as ST/SI/FL")
        print((ct / a.fps).round(0).astype(int).to_string(), "\n(seconds)")
        return acc

    report(f"child (id {a.child_id})", child, "child")
    adults = sorted(set(post["global_person_id"]) - {a.child_id})
    res = {pid: report(f"therapist tier vs id {pid}",
                       post[post["global_person_id"] == pid].set_index("frame")["cls"], "therapist")
           for pid in adults}
    if res:
        pid = max(res, key=lambda k: -1 if np.isnan(res[k]) else res[k])
        print(f"\n[elan] therapist tier best matches id {pid} ({res[pid]:.1%})")


if __name__ == "__main__":
    main()
