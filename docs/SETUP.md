# Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchreid gdown tensorboard  # OSNet -- see requirements.txt's notes
pip install mediapipe  # pose step only -- see pose/extract_keypoints.py
```

**`psifx` itself** (not a PyPI package):

```bash
git clone https://github.com/psifx/psifx
cd psifx && pip install .
```

SAM3 checkpoint access is gated (Meta requires ethical-approval access
via Hugging Face). To avoid psifx's automatic Hugging Face auth flow,
clone the SAM3 checkpoint locally and point `SAM3_PATH` in
`psifx/utils/constants.py` at it -- see psifx's own docs
(https://psifx.github.io/psifx/) for the exact steps and CUDA/PyTorch
requirements (needs a CUDA GPU; not runnable on a Mac).

The PoseLandmarker model (`.task` file, pose step) is a **public**
download, not gated like the SAM3 checkpoint -- `pose/model_cache.py`
fetches it itself into `<repo>/models/` (not `~/.cache/...`: a fixed
location inside the project, independent of the cwd the script is
launched from -- see that module's docstring for the real bug this
fixes) on first use. The URL is **pinned to a specific version**
(`float16/1/...`, not `.../latest/...`) for reproducibility across runs.
The default variant is `lite` (fastest; matches the legacy pipeline's
own choice -- pass `--model-variant full` or `heavy` for more accuracy
at the cost of speed). If the auto-download can't reach the internet
from wherever this runs, grab it by hand instead and pass `--model-path`:

```bash
curl -o pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

**`ffmpeg`** (system binary, `run_pipeline.py` shells out to it for
the transcode step): install via your OS package manager.

**`pyk4a`** (depth step only, Azure Kinect sessions -- see
[USAGE.md](USAGE.md#depth-after-pose-azure-kinect-sessions-only)):

```bash
pip install pyk4a
```

This alone is **not enough** -- `pyk4a` is a thin wrapper around the
real Azure Kinect Sensor SDK (`libk4a`), which has to be installed
system-wide separately. Azure Kinect DK is an officially discontinued
Microsoft product, so there's no current first-party install guide for
a modern Ubuntu -- expect to need a community guide (search "install
libk4a Ubuntu 22.04/24.04" or similar) and some trial and error;
`pip install pyk4a` failing to import (`ImportError` mentioning
`libk4a` or `.so`) means the SDK isn't found, not that `pyk4a` itself
is broken. **Not yet verified on a real machine** -- run
`pip install pyk4a && python -c "import pyk4a"` first and treat a
clean import as the actual go/no-go signal for the depth step, before
relying on anything below.
