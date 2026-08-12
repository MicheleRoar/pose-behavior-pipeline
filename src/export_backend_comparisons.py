"""
export_backend_comparisons.py
================================
Runs the SAME video through 4 fixed SAM 3.1 configurations (see CONFIGS
below, agreed with Michele 2026-08) and writes one annotated MP4 per
configuration -- ready to load side by side into the webui's "Compare
runs" window (see webui/api.py::Api.open_compare_window / compare.html)
to eyeball which combination keeps identities the most stable.

Unlike `benchmark_backends.py` (fast, no overlay, quantitative CSV only:
n_raw_ids/track lifespans/fps), this script goes through the REAL
pipeline (`gui/pipeline_runner.py::iter_pipeline_frames`, the exact same
function the webui GUI uses) so the output is a real annotated video, at
the cost of being slower. Use `benchmark_backends.py` first to get quick
numbers, this script to actually SEE the difference on the 1-2
configurations that look most promising (or, as here, on all 4 at once
if you want the full picture).

The 4 configurations (a 2x2: {box-mode w/o cross-chunk helpers, text-
prompt w/ cross-chunk helpers} x {no extra re-id layer, +OSNet re-id
layer}):

  1. sam_vanilla         -- SAM 3.1, box-mode (YOLO proposes boxes once
                             per chunk boundary, ids passed straight
                             through to SAM's obj_id -- see
                             ChunkedVideoPredictorBackend._seed_new_chunk's
                             docstring for why box mode never needs
                             geometric reconciliation in the first
                             place), appearance-gallery fallback OFF, no
                             SegReIdentifier. The true baseline: nothing
                             beyond what SAM itself does across chunks.
  2. sam_osnet            -- same base as (1), PLUS a SegReIdentifier
                             with the OSNet embedding on top (post-
                             processing layer, independent of SAM's own
                             logic -- see identity_gallery.py's docstring
                             in chunking.py for why this is a SEPARATE
                             system from SAM's internal one).
  3. sam_chunk_helpers    -- SAM 3.1, TEXT-PROMPT discovery (SAM finds
                             people on its own each chunk) with the
                             built-in chunk-boundary helpers ON: Hungarian
                             geometric reconciliation + motion
                             compensation (chunking.reconcile_ids_windowed)
                             + the internal appearance gallery
                             (identity_gallery.py). No extra re-id layer.
  4. sam_everything       -- (3) + the same OSNet SegReIdentifier layer
                             as (2), stacked on top.

Why box-mode for (1)/(2) and text-prompt for (3)/(4): the geometric
reconciliation helpers ONLY EVER run in text-prompt mode (box mode has
no local-id ambiguity to resolve, ids are passed to SAM directly) -- see
build_tracker()'s docstring in segmentation_demo.py. So "helpers off" and
"helpers on" aren't a single flag, they're a change of seeding strategy;
this is the closest fair comparison achievable without inventing a new
code path just for this benchmark.

`sam_redetect_every` is intentionally left unset (None) in ALL 4 configs:
combining it with text-prompt mode is known to make already-tracked
people flicker (see the warning in Sam31Tracker.__init__, discovered
2026-08 on the dancing-tracks test) -- not part of what's being compared
here.

Requires a CUDA GPU (SAM 3.1) and this project's optional 'torch'+
'torchreid' dependencies if any config with a re-id layer is selected
(2/4) -- see requirements.txt. Not runnable in the sandbox used to
develop the rest of this project; verify by hand on the real machine.

Usage:
    # all 4 configs, needs --max-people for (2) and (4) (SegReIdentifier
    # requires it, see gui/pipeline_runner.py)
    python export_backend_comparisons.py --source video.mp4 --fps 15 \\
        --max-people 2 --sam-chunk-size 300 --out-dir comparisons

    # only a subset
    python export_backend_comparisons.py --source video.mp4 --fps 15 \\
        --max-people 2 --sam-chunk-size 300 --out-dir comparisons \\
        --configs 1_sam_vanilla,3_sam_chunk_helpers
"""

from __future__ import annotations

import argparse
import os
import time

from common.device import detect_default_device
from common.video_writer import open_annotated_video_writer
from gui.pipeline_runner import iter_pipeline_frames

CONFIGS = [
    {
        "label": "1_sam_vanilla",
        "description": "SAM 3.1, box-mode, no cross-chunk helpers, no re-id layer (baseline).",
        "sam_text_prompt": None,
        "sam_appearance_fallback": False,
        "with_seg_reid": False,
        "use_appearance_embedding": False,
    },
    {
        "label": "2_sam_osnet",
        "description": "Same base as (1) + SegReIdentifier with OSNet embedding on top.",
        "sam_text_prompt": None,
        "sam_appearance_fallback": False,
        "with_seg_reid": True,
        "use_appearance_embedding": True,
    },
    {
        "label": "3_sam_chunk_helpers",
        "description": "SAM 3.1 text-prompt discovery + built-in chunk-boundary helpers "
                        "(Hungarian + motion compensation + internal appearance gallery).",
        "sam_text_prompt": "person",
        "sam_appearance_fallback": True,
        "with_seg_reid": False,
        "use_appearance_embedding": False,
    },
    {
        "label": "4_sam_everything",
        "description": "(3) + SegReIdentifier with OSNet embedding on top.",
        "sam_text_prompt": "person",
        "sam_appearance_fallback": True,
        "with_seg_reid": True,
        "use_appearance_embedding": True,
    },
]


def export_one(config: dict, *, source, fps: float, device: str,
                max_people: int | None, sam_chunk_size: int, sam_overlap: int,
                scale: str, conf_threshold: float, embedding_device: str,
                out_dir: str) -> str:
    """Runs ONE configuration end to end and writes its annotated video.
    Streams frame by frame straight to `cv2.VideoWriter` (not buffered in
    memory first) -- same reasoning as `Api.export_video()` in
    webui/api.py, just without a `VideoPlayer` cache in front of it since
    nothing here needs to support seeking/replay, only a single forward
    pass."""
    out_path = os.path.join(out_dir, f"{config['label']}.mp4")
    print(f"\n=== {config['label']}: {config['description']} ===")
    print(f"    -> {out_path}")

    writer = None
    codec = None
    n_frames = 0
    t_start = time.time()
    try:
        for frame in iter_pipeline_frames(
            mode="segmentation", source=source, fps=fps, device=device,
            seg_model=f"yolo26{scale}-seg.pt", seg_backend="sam31",
            sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
            sam_text_prompt=config["sam_text_prompt"],
            sam_appearance_fallback=config["sam_appearance_fallback"],
            with_seg_reid=config["with_seg_reid"],
            use_appearance_embedding=config["use_appearance_embedding"],
            embedding_device=embedding_device,
            max_people=max_people, conf_threshold=conf_threshold,
        ):
            if writer is None:
                height, width = frame.frame.shape[:2]
                writer, codec = open_annotated_video_writer(out_path, fps, width, height)
            writer.write(frame.frame)
            n_frames += 1
            if n_frames % 150 == 0:
                print(f"    ...{n_frames} frames ({n_frames / fps:.1f}s of video) processed")
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.time() - t_start
    fps_processed = n_frames / elapsed if elapsed > 0 else 0.0
    print(f"    done: {n_frames} frames written ({elapsed:.1f}s wall-clock, "
          f"{fps_processed:.1f} fps, codec={codec})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exports 4 fixed SAM 3.1 configurations of the same video as "
                     "annotated MP4s, for visual comparison in the webui's 'Compare "
                     "runs' window -- see the module docstring for exactly what each "
                     "of the 4 configs does.")
    parser.add_argument("--source", required=True, help="Video path")
    parser.add_argument("--fps", type=float, required=True, help="Source frame rate")
    parser.add_argument("--device", default=None,
                         help="Overrides auto-detection -- must resolve to 'cuda' "
                              "(SAM 3.1 requirement)")
    parser.add_argument("--max-people", type=int, default=None,
                         help="Known number of session participants -- REQUIRED for "
                              "configs 2/4 (2_sam_osnet, 4_sam_everything), which need "
                              "it to build a SegReIdentifier (see gui/pipeline_runner.py)")
    parser.add_argument("--sam-chunk-size", type=int, default=300,
                         help="Frames per SAM chunk -- same value used for all 4 "
                              "configs, for a fair comparison. Default: 300 (~20s at "
                              "15fps)")
    parser.add_argument("--sam-overlap", type=int, default=50)
    parser.add_argument("--scale", default="s", choices=["n", "s", "m"],
                         help="YOLO model size, used as the box proposer for SAM's "
                              "box-mode configs and by SegReIdentifier where relevant")
    parser.add_argument("--conf-threshold", type=float, default=0.1)
    parser.add_argument("--embedding-device", default="cpu",
                         help="Device for the OSNet embedder (configs 2/4) -- 'cpu' is "
                              "usually fine, it's a small model, independent of the "
                              "main --device used for SAM/YOLO")
    parser.add_argument("--out-dir", default="comparisons",
                         help="Output folder for the 4 (or fewer, see --configs) MP4s, "
                              "created if missing. Default: 'comparisons'")
    parser.add_argument("--configs", default=",".join(c["label"] for c in CONFIGS),
                         help=f"Comma-separated subset of config labels to run "
                              f"(default: all four -- {', '.join(c['label'] for c in CONFIGS)})")
    args = parser.parse_args()

    device = args.device or detect_default_device()
    if device != "cuda":
        raise SystemExit(
            f"All 4 configs use SAM 3.1, which requires device='cuda' "
            f"(detected {device!r}). Run this on the CUDA machine."
        )

    selected_labels = [c.strip() for c in args.configs.split(",") if c.strip()]
    known_labels = {c["label"] for c in CONFIGS}
    unknown = set(selected_labels) - known_labels
    if unknown:
        raise SystemExit(f"unknown config label(s): {sorted(unknown)} "
                          f"(expected among {sorted(known_labels)})")
    selected = [c for c in CONFIGS if c["label"] in selected_labels]

    needs_max_people = [c["label"] for c in selected if c["with_seg_reid"]]
    if needs_max_people and args.max_people is None:
        raise SystemExit(
            f"--max-people is required for: {needs_max_people} (SegReIdentifier "
            f"needs a known cap -- see gui/pipeline_runner.py)"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    written = [
        export_one(
            config, source=args.source, fps=args.fps, device=device,
            max_people=args.max_people, sam_chunk_size=args.sam_chunk_size,
            sam_overlap=args.sam_overlap, scale=args.scale,
            conf_threshold=args.conf_threshold, embedding_device=args.embedding_device,
            out_dir=args.out_dir,
        )
        for config in selected
    ]

    print(f"\nAll done -- {len(written)} video(s) written. Load these into the "
          f"'Compare runs' window (top-right icon in the main window):")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
