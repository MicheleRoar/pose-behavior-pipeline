"""
benchmark_backends.py
========================
Compares the available tracking backends (YOLO26-seg+heuristic ByteTrack,
SAM 3.1, SAM2 -- the latter two with or without reseeding of new people
at chunk boundaries, see segmentation/sam_backend.py) on the SAME video,
to understand which one keeps people's identity most stable over time.
Born from a concrete problem: during a therapeutic play session, children
continuously enter and leave the frame, or change clothes -- the tracking
id should stay the same.

Why SAM2 and not SAMURAI: SAMURAI was removed from the comparison, see
the docstring of `segmentation/sam2_estimation.py` -- its Kalman filter
assumes a single tracked object per session and crashes as soon as
multiple people are seeded together (the normal case here). Vanilla SAM2
supports multi-object natively.

No ground truth required: the metrics here are SELF-consistency ones,
not true IDF1/HOTA (which would require frame-by-frame labels of the
real identity, not currently available):

- how many "raw" identities (ByteTrack raw track_id or SAM equivalent)
  are created over the whole session -- if the number of real people is
  known (see --max-people) and the raw count is much higher, it's a sign
  the method loses and "reinvents" identities when someone leaves/
  re-enters or changes appearance;
- how long tracks last on average (min/median/max, in frames) and how
  many are "short" (below SHORT_LIVED_THRESHOLD_FRAMES) -- many short
  tracks indicate fragmentation: the same person gets split into
  multiple ids over time instead of staying one;
- processing time / fps -- the practical cost of each method, not just
  its quality.

If annotations emerge in the future (even just on individual events like
"child X leaves at frame N and returns at frame M"), the comparison can
be extended by adding a function dedicated to those specific events,
without touching the structure below.

Doesn't draw an overlay or open windows (faster, and the metrics above
don't require looking at the frames): uses `tracker.run()` directly (the
common `SegmentationBackend` protocol, see segmentation/backend.py), not
`iter_segmentation_frames()` (which draws the overlay, useless here).

If a method can't run on this machine (device other than "cuda" for
sam31/sam2, or a library not installed) it's SKIPPED with a warning
instead of failing the whole comparison -- useful for rerunning the same
command on different machines (Mac without CUDA: only "yolo" runs; a
CUDA machine with only SAM2 installed: the "sam31*" variants are
skipped).

chunk_size/overlap/redetect_every sweep
-------------------------------------------
Born from a concrete problem: which combination of these three
parameters (see segmentation/sam_backend.py::ChunkedVideoPredictorBackend)
works well for YOUR recordings (duration, how often children enter/
leave) can't be guessed on paper. `--sam-chunk-size`/`--sam-overlap`/
`--sam-redetect-every` NOW accept a comma-separated list instead of a
single value (e.g. `--sam-chunk-size 300,600`): the cartesian product of
all combinations is run automatically, ONLY ONCE per method if you don't
specify lists (original behavior, unchanged). The sweep only applies to
methods that actually use these parameters (`sam31`/`sam2*`) -- for
"yolo" (which ignores them) it runs once, not repeated for every
combination. An empty element between commas in `--sam-redetect-every`
(e.g. `100,`) also includes the "disabled" case in the same comparison.
Every CSV row reports the parameters used (`sam_chunk_size`/
`sam_overlap`/`sam_redetect_every`, `None` for "yolo" where not
relevant) and a readable `run_label`, so you can sort/filter by
`n_raw_ids`/`short_lived_ids_pct` and pick the best combination instead
of guessing.

Usage:
    python benchmark_backends.py --source video.mp4 --fps 15 \\
        --methods yolo,sam31,sam31-noreseed,sam2,sam2-noreseed \\
        --max-people 3 --out benchmark_results.csv

    # sweep: 2 chunk_size x 2 overlap x 3 redetect_every (including "off")
    # for sam31, in addition to a single yolo run -- 12 rows total
    python benchmark_backends.py --source video.mp4 --fps 15 \\
        --methods yolo,sam31 --sam-text-prompt person \\
        --sam-chunk-size 300,600 --sam-overlap 30,50 \\
        --sam-redetect-every 100,200, --out sweep.csv

    # "yolo vanilla" vs "sam31 chunked, no helpers" vs "sam31 chunked +
    # helpers" (geometric reconciliation + appearance gallery, only
    # engaged in text-prompt mode, see build_tracker()'s docstring in
    # segmentation_demo.py) -- run separately (different --sam-text-prompt/
    # --sam-appearance-fallback need their own invocation, can't mix
    # within one --methods list) and compare the resulting CSVs:
    python benchmark_backends.py --source video.mp4 --fps 15 --methods yolo \\
        --out run1_yolo.csv
    python benchmark_backends.py --source video.mp4 --fps 15 --methods sam31 \\
        --sam-chunk-size 300 --no-sam-appearance-fallback --out run2_sam_nohelpers.csv
    python benchmark_backends.py --source video.mp4 --fps 15 --methods sam31 \\
        --sam-chunk-size 300 --sam-text-prompt person --out run3_sam_helpers.csv
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict

import pandas as pd

from common.device import detect_default_device
from segmentation_demo import build_tracker

METHOD_PRESETS = {
    "yolo": dict(backend="yolo", reseed=True),
    "sam31": dict(backend="sam31", reseed=True),
    "sam31-noreseed": dict(backend="sam31", reseed=False),
    "sam2": dict(backend="sam2", reseed=True),
    "sam2-noreseed": dict(backend="sam2", reseed=False),
}

# Same reference used elsewhere in the project (reid.py::min_signature_frames,
# run_segmentation()/track_stability_check.py) -- comparable with those
# already-familiar statistics, not a number invented just for here.
SHORT_LIVED_THRESHOLD_FRAMES = 15


def run_one_method(method: str, *, source, fps: float, device: str,
                    model_scale: str = "s", conf_threshold: float = 0.1,
                    tracker_config: str = "bytetrack.yaml",
                    max_people: int | None = None,
                    sam_chunk_size: int = 600, sam_overlap: int = 50,
                    sam_redetect_every: int | None = None,
                    sam_text_prompt: str | None = None,
                    sam_appearance_fallback: bool = True) -> dict | None:
    """Runs ONE method on the video and returns a metrics dict, or `None`
    if the method must be skipped (incompatible device or missing
    library -- see the module docstring). Never raises for either of
    these two expected reasons, only for a real bug (e.g. an unknown
    parameter elsewhere)."""
    preset = METHOD_PRESETS[method]
    backend = preset["backend"]
    reseed = preset["reseed"]

    if backend in ("sam31", "sam2") and device != "cuda":
        print(f"[{method}] skipped: requires device='cuda' (detected {device!r})")
        return None

    try:
        tracker = build_tracker(
            backend, model_name=f"yolo26{model_scale}-seg.pt", device=device,
            conf_threshold=conf_threshold, tracker_config=tracker_config,
            max_people=max_people, sam_chunk_size=sam_chunk_size,
            sam_overlap=sam_overlap, sam_chunk_store_dir=None,
            sam_reseed_new_people=reseed,
            sam_redetect_every=sam_redetect_every, sam_text_prompt=sam_text_prompt,
            sam_appearance_fallback=sam_appearance_fallback,
        )
    except ImportError as exc:
        print(f"[{method}] skipped: {exc}")
        return None

    raw_id_frame_count: dict[int, int] = defaultdict(int)
    n_frames = 0
    t_start = time.time()
    for frame_result in tracker.run(source=source):
        n_frames = frame_result.frame_index + 1
        for track_id, _bbox, _poly, _conf in frame_result.people:
            raw_id_frame_count[track_id] += 1
    elapsed_s = time.time() - t_start

    n_ids = len(raw_id_frame_count)
    lifespans = sorted(raw_id_frame_count.values())
    short_lived = sum(1 for v in lifespans if v < SHORT_LIVED_THRESHOLD_FRAMES)
    median_frames = lifespans[len(lifespans) // 2] if lifespans else 0

    # Reported in the output even for a single run (not just in a sweep):
    # None for "yolo", which receives them but ignores them (SegTracker
    # doesn't use them) -- so the CSV doesn't imply they were applied.
    sam_params_relevant = backend != "yolo"
    return {
        "method": method,
        "backend": backend,
        "reseed_new_people": reseed,
        "sam_chunk_size": sam_chunk_size if sam_params_relevant else None,
        "sam_overlap": sam_overlap if sam_params_relevant else None,
        "sam_redetect_every": sam_redetect_every if sam_params_relevant else None,
        "sam_text_prompt": sam_text_prompt if sam_params_relevant else None,
        "sam_appearance_fallback": sam_appearance_fallback if sam_params_relevant else None,
        "n_frames": n_frames,
        "n_raw_ids": n_ids,
        "lifespan_min_frames": lifespans[0] if lifespans else 0,
        "lifespan_median_frames": median_frames,
        "lifespan_median_s": round(median_frames / fps, 2) if fps > 0 else 0.0,
        "lifespan_max_frames": lifespans[-1] if lifespans else 0,
        "short_lived_ids_pct": round(100 * short_lived / n_ids, 1) if n_ids else 0.0,
        # elapsed_s/processing_fps: COMPUTATION TIME (wall-clock to
        # process the video), not to be confused with lifespan_median_s
        # (a track's average duration on the video's TIMELINE, based on
        # the source --fps).
        "elapsed_s": round(elapsed_s, 1),
        "processing_fps": round(n_frames / elapsed_s, 2) if elapsed_s > 0 else 0.0,
    }


def run_benchmark(methods: list[str], *, source, fps: float, device: str | None = None,
                   sam_chunk_sizes: list[int] = (600,), sam_overlaps: list[int] = (50,),
                   sam_redetect_everys: list[int | None] = (None,),
                   **kwargs) -> pd.DataFrame:
    """Runs all `methods` (in the given order) on the same `source` and
    returns a DataFrame with one row per NON-skipped combination. Missing
    column for a skipped method: simply absent from the result, not a row
    with null values -- the caller immediately sees how many/which
    methods actually ran.

    `sam_chunk_sizes`/`sam_overlaps`/`sam_redetect_everys` (lists,
    default a single value each -- original behavior unchanged if not
    specified): for methods that actually use these parameters (`backend`
    != "yolo") the cartesian product of all combinations is run, see the
    module docstring for why (no way to know in advance which
    combination works for a specific recording). For "yolo" (which
    ignores them) a SINGLE run is executed, not repeated for every
    combination -- it would waste time on identical rows."""
    device = device or detect_default_device()
    rows = []
    for method in methods:
        if method not in METHOD_PRESETS:
            raise ValueError(f"unknown method: {method!r} (expected one of {sorted(METHOD_PRESETS)})")
        backend = METHOD_PRESETS[method]["backend"]
        sweeping = backend != "yolo"
        combos = (
            list(itertools.product(sam_chunk_sizes, sam_overlaps, sam_redetect_everys))
            if sweeping else [(sam_chunk_sizes[0], sam_overlaps[0], sam_redetect_everys[0])]
        )
        multi_combo = sweeping and len(combos) > 1
        for chunk_size, overlap, redetect_every in combos:
            if sweeping and chunk_size <= overlap:
                print(f"[{method}] skipped invalid combination: "
                      f"chunk_size={chunk_size} <= overlap={overlap}")
                continue
            label = method
            if multi_combo:
                label = f"{method}[cs={chunk_size},ov={overlap},rd={redetect_every}]"
            print(f"--- {label} ---")
            result = run_one_method(
                method, source=source, fps=fps, device=device,
                sam_chunk_size=chunk_size, sam_overlap=overlap, sam_redetect_every=redetect_every,
                **kwargs,
            )
            if result is not None:
                result["run_label"] = label
                rows.append(result)
    return pd.DataFrame(rows)


def _parse_int_list(raw: str, *, allow_none: bool = False) -> list[int | None]:
    """'600,300' -> [600, 300]. An empty element between commas (e.g.
    '100,') becomes `None` if `allow_none=True` -- used to include the
    "disabled" case (redetect_every=None) in the same sweep instead of
    having to launch a separate command. Empty/all-empty string ->
    `[None]` if `allow_none`, otherwise an empty list (a downstream
    error, a parameter without `allow_none` must have at least one
    value)."""
    values: list[int | None] = []
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            if allow_none:
                values.append(None)
            continue
        values.append(int(part))
    if not values and allow_none:
        values = [None]
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Compares tracking backends (YOLO/SAM 3.1/SAM2, with/without "
                     "reseeding of new people) on the same video: how many 'raw' "
                     "identities each one creates, how long tracks last, how fast it is. "
                     "No ground truth required -- see the module docstring.")
    parser.add_argument("--source", required=True, help="Video path")
    parser.add_argument("--fps", type=float, required=True,
                         help="Source frame rate -- used to convert the median track "
                              "duration from frames to seconds (lifespan_median_s)")
    parser.add_argument("--methods", default=",".join(METHOD_PRESETS),
                         help=f"Comma-separated list among {sorted(METHOD_PRESETS)} "
                              f"(default: all)")
    parser.add_argument("--device", default=None,
                         help="Overrides auto-detection (cuda/mps/cpu)")
    parser.add_argument("--scale", default="s", choices=["n", "s", "m"],
                         help="YOLO model size (used as the tracker for 'yolo', "
                              "and as the prompt proposer for sam31/samurai)")
    parser.add_argument("--conf-threshold", type=float, default=0.1)
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="ByteTrack config (only for the 'yolo' method)")
    parser.add_argument("--max-people", type=int, default=None,
                         help="Known number of session participants, if you know it -- "
                              "used as a cap for YOLO and to interpret n_raw_ids "
                              "(much higher than the real number = lost/reinvented identities)")
    parser.add_argument("--sam-chunk-size", default="600",
                         help="One or more comma-separated values (e.g. '300,600') -- with "
                              "more than one value, the sweep runs over all combinations "
                              "with --sam-overlap/--sam-redetect-every (sam31/sam2 only, see "
                              "the module docstring). Default: '600' (a single run, as before)")
    parser.add_argument("--sam-overlap", default="50",
                         help="Like --sam-chunk-size, one or more comma-separated values. "
                              "Default: '50'")
    parser.add_argument("--sam-redetect-every", default="",
                         help="One or more comma-separated values (e.g. '100,200'). An "
                              "empty element between commas (e.g. '100,') also includes the "
                              "'disabled' case in the same sweep. Default: '' (disabled, a "
                              "single run, as before) -- reruns YOLO every N frames inside "
                              "the chunk instead of only at the start, see "
                              "ChunkedVideoPredictorBackend in segmentation/sam_backend.py")
    parser.add_argument("--sam-text-prompt", default=None,
                         help="Text prompt for people discovery via SAM 3.1 "
                              "(e.g. 'person'), independent from YOLO -- ignored for other methods")
    parser.add_argument("--sam-appearance-fallback", action=argparse.BooleanOptionalAction, default=True,
                         help="OSNet appearance-based fallback for people who can't be "
                              "geometrically matched across a chunk boundary (sam31/sam2 "
                              "only, see identity_gallery.py). Default: on. Use "
                              "--no-sam-appearance-fallback together with a run that has NO "
                              "--sam-text-prompt to get a true 'chunked, zero cross-chunk "
                              "helpers' baseline -- geometric reconciliation (Hungarian + "
                              "motion compensation) only runs in text-prompt mode to begin "
                              "with, see build_tracker()'s docstring in segmentation_demo.py.")
    parser.add_argument("--out", default="benchmark_results.csv")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sam_chunk_sizes = _parse_int_list(args.sam_chunk_size)
    sam_overlaps = _parse_int_list(args.sam_overlap)
    sam_redetect_everys = _parse_int_list(args.sam_redetect_every, allow_none=True)
    df = run_benchmark(
        methods, source=args.source, fps=args.fps, device=args.device,
        model_scale=args.scale, conf_threshold=args.conf_threshold,
        tracker_config=args.tracker, max_people=args.max_people,
        sam_chunk_sizes=sam_chunk_sizes, sam_overlaps=sam_overlaps,
        sam_redetect_everys=sam_redetect_everys, sam_text_prompt=args.sam_text_prompt,
        sam_appearance_fallback=args.sam_appearance_fallback,
    )
    if df.empty:
        print("No method ran (all skipped) -- nothing to save.")
        return
    df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}:\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
