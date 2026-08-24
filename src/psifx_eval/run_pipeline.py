"""
psifx_eval/run_pipeline.py
=============================
Single entry point chaining the 4 steps Michele otherwise ran by hand
(ffmpeg trim -> `run_sam3_baseline` -> `merge_fragments` -> psifx's own
`TrackingTool.visualize` overlay) into one resumable run.

Everything lives next to the source video, inside three top-level
subfolders of the video's own directory (not a separate output root):

    <video_dir>/
        processed/<name>.mp4          # always: ffmpeg re-encode (trimmed if --ss/--to given, full video otherwise)
        masks/<name>/                 # raw sam3_baseline MaskDir
        merged/<name>/
            <id>.mp4 ...              # merged MaskDir (merge_fragments)
            merge_report.json
            overlay.mp4                # TrackingTool.visualize output

`<name>` is the video's filename stem, plus an exact range suffix when
`--ss`/`--to` are given (digits only, taken directly from the two
timestamps -- e.g. `--ss 00:22:34 --to 00:27:40` -> `_002234-002740`),
so a whole-video run and any number of ranged runs on the same source
video coexist as sibling folders under masks/ and merged/ without ever
colliding.

The `processed/` re-encode step ALWAYS runs, even with no range -- it's
normalization (`-c:v libx264 -pix_fmt yuv420p`), not just a trim. Source
camera `.mkv` files can use a codec/pixel format cv2 and psifx's video
reading don't decode cleanly (corrupted "noise" frames, no error
raised), so `masks/`/`merged/` always read from `processed/`, never
from the original file.

Each step is skipped if its output already exists (resumable -- SAM3
can take a long time, no reason to redo it because a later step
failed) unless `--overwrite` is passed, which forces every step to
re-run.

Usage:
    python -m psifx_eval.run_pipeline --video ~/Bureau/9_group_1_3/camera_a.mkv \\
        --ss 00:22:34 --to 00:27:40 --device cuda

    # whole video (still re-encoded to processed/camera_a.mp4, just not trimmed):
    python -m psifx_eval.run_pipeline --video ~/Bureau/9_individual_58/camera_a.mkv
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def _sanitize_timestamp(ts: str) -> str:
    """Digits only, e.g. '00:22:34' -> '002234', '00:22:34.5' -> '0022345'.
    Deterministic and collision-free by construction: two different
    --ss/--to strings only produce the same suffix if they were the same
    string with different punctuation, which never happens for ffmpeg's
    own HH:MM:SS[.ms] / seconds syntax."""
    return re.sub(r"[^0-9]", "", ts)


def resolve_name(video_path: str, ss: str | None, to: str | None) -> str:
    """The `<name>` used for every output folder/file -- video stem,
    plus an exact range suffix when a range is given. See module
    docstring for why this makes whole-video and ranged runs coexist
    without collisions."""
    stem = Path(video_path).stem
    if ss is None and to is None:
        return stem
    return f"{stem}_{_sanitize_timestamp(ss)}-{_sanitize_timestamp(to)}"


def resolve_paths(video_path: str, ss: str | None, to: str | None) -> dict[str, Path]:
    """All the folder/file paths for one pipeline run, all rooted at
    the source video's own directory -- see module docstring."""
    video_path = Path(video_path)
    video_dir = video_path.parent
    name = resolve_name(str(video_path), ss, to)

    paths = {
        "video_dir": video_dir,
        "masks_dir": video_dir / "masks" / name,
        "merged_dir": video_dir / "merged" / name,
    }
    paths["merge_report"] = paths["merged_dir"] / "merge_report.json"
    paths["overlay"] = paths["merged_dir"] / "overlay.mp4"

    paths["processed_clip"] = video_dir / "processed" / f"{name}.mp4"

    return paths


def _transcode(video_path: str, ss: str | None, to: str | None, out_path: Path, overwrite: bool) -> str:
    """ffmpeg re-encode to libx264/yuv420p -- ALWAYS runs, trimmed to
    `[ss, to]` if given, full video otherwise. Same flags Michele already
    used by hand. Not optional: normalizes source `.mkv` files whose
    codec/pixel format cv2/psifx don't decode cleanly on their own (see
    module docstring). Returns the path actually used as input to the
    rest of the pipeline."""
    if out_path.exists() and not overwrite:
        print(f"[run_pipeline] processed video already exists, skipping re-encode -> {out_path}")
        return str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y" if overwrite else "-n", "-i", video_path]
    if ss is not None:
        cmd += ["-ss", ss, "-to", to]
    cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", str(out_path)]
    label = f"[{ss} -> {to}]" if ss is not None else "(whole video)"
    print(f"[run_pipeline] re-encoding {video_path} {label} -> {out_path}")
    subprocess.run(cmd, check=True)
    return str(out_path)


def _has_mask_files(mask_dir: Path) -> bool:
    return mask_dir.is_dir() and any(mask_dir.glob("*.mp4"))


def run_pipeline(
    *,
    video_path: str,
    ss: str | None = None,
    to: str | None = None,
    device: str = "cuda",
    visualize_device: str = "cpu",
    overwrite: bool = False,
    # run_sam3_baseline pass-through (defaults match run_sam3_baseline.py)
    chunk_size: int = 400,
    text_prompt: str = "person",
    iou_threshold: float = 0.15,
    max_objects: int | None = None,
    model_path: str | None = None,
    # merge_fragments pass-through (defaults match merge_fragments.py)
    min_fragment_frames: int | None = None,
    merge_threshold: float | None = None,
    signature_samples: int | None = None,
    pooled_group_samples: int | None = None,
    use_osnet: bool = True,
) -> dict[str, str]:
    """Runs the 4-step pipeline for one video (optionally trimmed to
    `[ss, to]` first), resuming past any step whose output already
    exists unless `overwrite=True`. Returns the resolved output paths.
    See module docstring for the folder layout and naming."""
    if (ss is None) != (to is None):
        raise ValueError("--ss and --to must be given together, or not at all")

    paths = resolve_paths(video_path, ss, to)

    # Step 1: re-encode/normalize (always -- see module docstring on why
    # this can't be conditional on a range being given)
    sam3_input = _transcode(video_path, ss, to, paths["processed_clip"], overwrite)

    # Step 2: run_sam3_baseline -> masks/<name>/
    if _has_mask_files(paths["masks_dir"]) and not overwrite:
        print(f"[run_pipeline] masks already exist, skipping SAM3 baseline -> {paths['masks_dir']}")
    else:
        from psifx_eval.run_sam3_baseline import run_sam3_baseline

        paths["masks_dir"].mkdir(parents=True, exist_ok=True)
        print(f"[run_pipeline] running SAM3 baseline -> {paths['masks_dir']}")
        run_sam3_baseline(
            video_path=sam3_input, mask_dir=str(paths["masks_dir"]),
            chunk_size=chunk_size, text_prompt=text_prompt, iou_threshold=iou_threshold,
            max_objects=max_objects, model_path=model_path, device=device, overwrite=overwrite,
        )

    # Step 3: merge_fragments -> merged/<name>/
    if paths["merge_report"].exists() and not overwrite:
        print(f"[run_pipeline] merged masks already exist, skipping merge -> {paths['merged_dir']}")
    else:
        from psifx_eval.merge_fragments import (
            DEFAULT_MERGE_THRESHOLD, DEFAULT_MIN_FRAGMENT_FRAMES,
            DEFAULT_POOLED_SAMPLES_PER_MEMBER, DEFAULT_SIGNATURE_SAMPLES, merge_fragments,
        )
        import json

        paths["merged_dir"].mkdir(parents=True, exist_ok=True)
        print(f"[run_pipeline] merging fragments -> {paths['merged_dir']}")
        report = merge_fragments(
            video_path=sam3_input, mask_dir=str(paths["masks_dir"]), out_mask_dir=str(paths["merged_dir"]),
            min_fragment_frames=min_fragment_frames if min_fragment_frames is not None else DEFAULT_MIN_FRAGMENT_FRAMES,
            merge_threshold=merge_threshold if merge_threshold is not None else DEFAULT_MERGE_THRESHOLD,
            signature_samples=signature_samples if signature_samples is not None else DEFAULT_SIGNATURE_SAMPLES,
            pooled_samples_per_member=pooled_group_samples if pooled_group_samples is not None else DEFAULT_POOLED_SAMPLES_PER_MEMBER,
            device=device, use_osnet=use_osnet,
        )
        with open(paths["merge_report"], "w") as f:
            json.dump(report, f, indent=2)

    # Step 4: overlay -> merged/<name>/overlay.mp4 (psifx's own TrackingTool.visualize,
    # same call Michele already used by hand -- see module docstring)
    if paths["overlay"].exists() and not overwrite:
        print(f"[run_pipeline] overlay already exists, skipping -> {paths['overlay']}")
    else:
        from psifx.video.tracking.tool import TrackingTool

        mask_paths = sorted(paths["merged_dir"].glob("*.mp4"))
        tool = TrackingTool(device=visualize_device, overwrite=overwrite, verbose=True)
        print(f"[run_pipeline] rendering overlay -> {paths['overlay']}")
        tool.visualize(
            video_path=Path(sam3_input), mask_paths=mask_paths,
            visualization_path=paths["overlay"],
            blackout=False, color=True, labels=True,
        )

    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs the full pipeline (optional trim -> SAM3 baseline -> merge_fragments -> "
                     "overlay) for one video in one resumable pass -- see module docstring.")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--ss", default=None, help="Trim start (ffmpeg timestamp, e.g. 00:22:34) -- requires --to")
    parser.add_argument("--to", default=None, help="Trim end (ffmpeg timestamp, e.g. 00:27:40) -- requires --ss")
    parser.add_argument("--device", default="cuda", help="Device for SAM3 baseline + merge_fragments' OSNet embedder")
    parser.add_argument("--visualize-device", default="cpu", help="Device for the final overlay render")
    parser.add_argument("--overwrite", action="store_true", help="Force every step to re-run, even if its output already exists")

    sam3 = parser.add_argument_group("SAM3 baseline")
    sam3.add_argument("--chunk-size", type=int, default=400)
    sam3.add_argument("--text-prompt", default="person")
    sam3.add_argument("--iou-threshold", type=float, default=0.15)
    sam3.add_argument("--max-objects", type=int, default=None)
    sam3.add_argument("--model-path", default=None)

    merge = parser.add_argument_group("merge_fragments")
    merge.add_argument("--min-fragment-frames", type=int, default=None)
    merge.add_argument("--merge-threshold", type=float, default=None)
    merge.add_argument("--signature-samples", type=int, default=None)
    merge.add_argument("--pooled-group-samples", type=int, default=None)
    merge.add_argument("--no-osnet", dest="use_osnet", action="store_false", default=True)

    args = parser.parse_args()
    if (args.ss is None) != (args.to is None):
        parser.error("--ss and --to must be given together")

    result = run_pipeline(
        video_path=args.video, ss=args.ss, to=args.to,
        device=args.device, visualize_device=args.visualize_device, overwrite=args.overwrite,
        chunk_size=args.chunk_size, text_prompt=args.text_prompt, iou_threshold=args.iou_threshold,
        max_objects=args.max_objects, model_path=args.model_path,
        min_fragment_frames=args.min_fragment_frames, merge_threshold=args.merge_threshold,
        signature_samples=args.signature_samples, pooled_group_samples=args.pooled_group_samples,
        use_osnet=args.use_osnet,
    )
    print("\nDone:")
    for key, path in result.items():
        if key != "video_dir":
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
