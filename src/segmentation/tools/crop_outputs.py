"""
segmentation/tools/crop_outputs.py
====================================
Post-processing step for run_pipeline.py (step 5, after merge_fragments
and after the overlay of step 4): crops the merged MaskDir + its source
video to exclude camera_a's border artifact (left/top corner artifact,
~20px/~16px measured, margin chosen with a safety buffer, symmetric on
all 4 sides to keep proportions).

Does NOT reimplement cropping: it's a thin loop around psifx's own
`psifx.video.manipulation.tool.ManipulationTool`, the same tool exposed
by the `psifx video manipulation process` CLI command -- EXCEPT for
inputs with no audio stream (this pipeline's videos: Azure Kinect
recordings, video-only), where ManipulationTool.process() hard-fails
(`ffmpeg.output(video, audio, ...)` unconditionally maps an audio
stream that doesn't exist -- "Stream map '0:a' matches no streams").
For those we fall back to the exact same ffmpeg-python crop filter,
just without mapping an audio stream. Never touches
merge_fragments.py's fusion logic, and never touches the un-cropped
merged/<name>/ output -- writes to a separate sibling directory.

Output layout (mirrors the MaskDir convention from mask_io.py, so the
result can be fed straight into `TrackingTool.visualize` exactly like
step 4 of run_pipeline.py does for the un-cropped version):

    <out_dir>/
        <id>.mp4             one per id, cropped (same ids as mask_dir)
        _source/<stem>_cropped.mp4   cropped copy of the source video

Usage (standalone):
    python -m segmentation.tools.crop_outputs \\
        --video /path/processed/camera_a.mp4 \\
        --mask-dir /path/merged/camera_a \\
        --out-dir /path/merged/camera_a_cropped25 \\
        --margin 25
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import cv2
import ffmpeg

from psifx.video.manipulation.tool import ManipulationTool

_MASK_FILENAME_RE = re.compile(r"^(\d+)\.mp4$")


def _video_resolution(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def _has_audio_stream(path: Path) -> bool:
    """True if `path` has at least one audio stream (ffprobe)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())


def _crop_no_audio(in_path: Path, out_path: Path, x_min: int, y_min: int, x_max: int, y_max: int,
                    overwrite: bool, verbose: bool):
    """The exact same crop as ManipulationTool.process() (same
    ffmpeg-python library, same filter), just without mapping an audio
    stream -- for files that don't have one (see module docstring)."""
    if out_path.exists():
        if overwrite:
            out_path.unlink()
        else:
            raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video = ffmpeg.input(str(in_path)).video.crop(
        x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min,
    )
    output = ffmpeg.output(video, str(out_path))
    try:
        output.overwrite_output().run(quiet=not verbose)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr)
        print(f"\n[crop_outputs] ffmpeg stderr (no-audio path) for {in_path}:\n{stderr[-4000:]}\n", flush=True)
        raise


def crop_mask_dir(
    video_path: str | Path,
    mask_dir: str | Path,
    out_dir: str | Path,
    margin: int = 25,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, str]:
    """Crops `video_path` and every `<id>.mp4` in `mask_dir` by `margin`
    pixels on all 4 sides (true crop, not zeroing out -- the resulting
    frame is smaller), writing a fresh MaskDir + source video into
    `out_dir`. Skips a file if its cropped output already exists and
    `overwrite` is False (same resumability convention as the rest of
    run_pipeline.py).

    Returns {"out_dir": ..., "source_cropped": ...}.
    """
    video_path = Path(video_path)
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)
    source_dir = out_dir / "_source"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(
        (p for p in mask_dir.iterdir() if _MASK_FILENAME_RE.match(p.name)),
        key=lambda p: int(_MASK_FILENAME_RE.match(p.name).group(1)),
    )
    if not mask_files:
        raise ValueError(f"No '<id>.mp4' files found in {mask_dir} -- not a valid MaskDir.")

    tool = ManipulationTool(overwrite=overwrite, verbose=verbose)

    def _crop_one(in_path: Path, out_path: Path):
        if out_path.exists() and not overwrite:
            if verbose:
                print(f"[crop_outputs] already exists, skipping -> {out_path}")
            return
        width, height = _video_resolution(in_path)
        x_min, y_min = margin, margin
        x_max, y_max = width - margin, height - margin
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Margin {margin}px too large for {in_path} ({width}x{height})")

        if _has_audio_stream(in_path):
            try:
                tool.process(
                    in_video_path=in_path, out_video_path=out_path,
                    x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                )
            except ffmpeg.Error as e:
                stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr)
                print(f"\n[crop_outputs] ffmpeg stderr for {in_path}:\n{stderr[-4000:]}\n", flush=True)
                raise
        else:
            if verbose:
                print(f"[crop_outputs] {in_path.name} has no audio stream, cropping without mapping audio")
            _crop_no_audio(in_path, out_path, x_min, y_min, x_max, y_max, overwrite, verbose)

    source_cropped = source_dir / f"{video_path.stem}_cropped.mp4"
    _crop_one(video_path, source_cropped)

    for mf in mask_files:
        _crop_one(mf, out_dir / mf.name)

    return {"out_dir": str(out_dir), "source_cropped": str(source_cropped)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="source video (already processed, e.g. processed/camera_a.mp4)")
    parser.add_argument("--mask-dir", required=True, help="merge_fragments.py MaskDir (flat <id>.mp4 folder)")
    parser.add_argument("--out-dir", required=True, help="output directory (new cropped MaskDir)")
    parser.add_argument("--margin", type=int, default=25, help="pixels to crop on each side (default 25)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = crop_mask_dir(
        video_path=args.video, mask_dir=args.mask_dir, out_dir=args.out_dir,
        margin=args.margin, overwrite=args.overwrite,
    )
    print("\nDone:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
