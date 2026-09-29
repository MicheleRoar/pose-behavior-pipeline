"""
device.py
==========
Autodetection of the best available compute device (CUDA > MPS > CPU), so
there's no need to manually set `--device`/the GUI device on every
different machine. Before this module, `device="mps"` was hardcoded in
both GUIs (gui/app.py, webui/api.py + webui/app.js) and as the argparse
default in every CLI: on a PC with an NVIDIA GPU (CUDA, no Metal) this made
the pipeline silently fail instead of using the available GPU.

The CLIs still keep `--device` as an explicit override -- this module only
provides the DEFAULT when the user doesn't specify anything.
"""

from __future__ import annotations


def detect_default_device() -> str:
    """"cuda" if an NVIDIA GPU is available via PyTorch, otherwise "mps" if
    Apple Silicon's Metal device is available, otherwise "cpu". torch is
    imported inside the function (not at the top of the module), for
    consistency with the "delayed heavy import" style already used
    elsewhere in the project for heavy dependencies -- even though torch is
    still a mandatory indirect dependency here (it comes with ultralytics),
    so the import never actually fails in practice."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"
