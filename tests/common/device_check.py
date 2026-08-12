"""
device_check.py
=================
Verifies the logic of `common/device.py::detect_default_device()` WITHOUT
requiring torch installed or a real GPU: injects a fake `torch` module
into `sys.modules` with only the attributes the function reads
(`torch.cuda.is_available()`, `torch.backends.mps.is_available()`), so
every combination can be checked (CUDA present, MPS only, no GPU, MPS
backend absent on old torch versions) without needing real hardware --
useful because this pipeline's development environment has neither CUDA
nor torch installed (only the other, lighter dependencies), but the
behavior still needs to be guaranteed correct on any machine where the
pipeline actually runs (Apple Silicon Mac, PC with an NVIDIA GPU, or a
machine with no GPU).

Run with: python device_check.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def _install_fake_torch(*, cuda_available: bool, mps_available: bool | None):
    """Injects a fake `torch` into sys.modules with only what
    detect_default_device() reads. `mps_available=None` simulates a torch
    version without `torch.backends.mps` (old versions), to verify that
    getattr(..., None) doesn't make the function blow up."""
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake_torch.cuda = fake_cuda

    if mps_available is None:
        fake_backends = types.SimpleNamespace()  # no "mps" attribute
    else:
        fake_backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available)
        )
    fake_torch.backends = fake_backends

    sys.modules["torch"] = fake_torch
    # detect_default_device() imports torch inside the function: we also
    # need to invalidate any already-imported/cached device.py module to
    # make sure the next import picks up the fake torch.
    sys.modules.pop("common.device", None)


def part1_cuda_available_wins_over_mps():
    _install_fake_torch(cuda_available=True, mps_available=True)
    from common.device import detect_default_device
    assert detect_default_device() == "cuda"
    print("Part 1: CUDA available -> 'cuda' (takes priority even if MPS is also available) — OK")


def part2_mps_used_when_cuda_unavailable():
    _install_fake_torch(cuda_available=False, mps_available=True)
    from common.device import detect_default_device
    assert detect_default_device() == "mps"
    print("Part 2: no CUDA, MPS available -> 'mps' (Apple Silicon Mac case) — OK")


def part3_cpu_when_neither_available():
    _install_fake_torch(cuda_available=False, mps_available=False)
    from common.device import detect_default_device
    assert detect_default_device() == "cpu"
    print("Part 3: neither CUDA nor MPS available -> 'cpu' (final fallback) — OK")


def part4_missing_mps_backend_attribute_falls_back_to_cpu_not_crash():
    """On a torch version without `torch.backends.mps` (roughly pre-1.12)
    the function must not raise AttributeError -- it must treat it as
    "MPS unavailable" and fall back to cpu (or cuda if available)."""
    _install_fake_torch(cuda_available=False, mps_available=None)
    from common.device import detect_default_device
    assert detect_default_device() == "cpu"
    print("Part 4: a torch without torch.backends.mps (old versions) doesn't make the "
          "function blow up, falls back to 'cpu' instead of raising AttributeError — OK")


def main():
    part1_cuda_available_wins_over_mps()
    part2_mps_used_when_cuda_unavailable()
    part3_cpu_when_neither_available()
    part4_missing_mps_backend_attribute_falls_back_to_cpu_not_crash()
    print("\nVerification completed with no errors: detect_default_device() picks CUDA > MPS > CPU "
          "in every combination, without requiring torch/a real GPU in this environment.")


if __name__ == "__main__":
    main()
