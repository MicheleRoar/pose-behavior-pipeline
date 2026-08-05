"""
device_check.py
=================
Verifica della logica di `common/device.py::detect_default_device()` SENZA
richiedere torch installato ne' una GPU vera: inietta un modulo `torch`
finto in `sys.modules` con solo gli attributi letti dalla funzione
(`torch.cuda.is_available()`, `torch.backends.mps.is_available()`), cosi'
si puo' controllare ogni combinazione (CUDA presente, solo MPS, nessuna
GPU, backend MPS assente su versioni vecchie di torch) senza bisogno
dell'hardware reale -- utile perche' l'ambiente di sviluppo di questa
pipeline non ha ne' CUDA ne' torch installato (solo le altre dipendenze
piu' leggere), ma il comportamento va comunque garantito corretto su
qualunque macchina dove la pipeline gira davvero (Mac Apple Silicon, PC
con GPU NVIDIA, o macchina senza GPU).

Esegui con: python device_check.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _install_fake_torch(*, cuda_available: bool, mps_available: bool | None):
    """Inietta in sys.modules un `torch` finto con solo cio' che
    detect_default_device() legge. `mps_available=None` simula una
    versione di torch senza `torch.backends.mps` (versioni vecchie), per
    verificare che getattr(..., None) non faccia esplodere la funzione."""
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake_torch.cuda = fake_cuda

    if mps_available is None:
        fake_backends = types.SimpleNamespace()  # niente attributo "mps"
    else:
        fake_backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available)
        )
    fake_torch.backends = fake_backends

    sys.modules["torch"] = fake_torch
    # detect_default_device() importa torch dentro la funzione: bisogna
    # anche invalidare l'eventuale modulo device.py gia' importato/cache
    # per essere sicuri che il prossimo import prenda il torch finto.
    sys.modules.pop("common.device", None)


def part1_cuda_available_wins_over_mps():
    _install_fake_torch(cuda_available=True, mps_available=True)
    from common.device import detect_default_device
    assert detect_default_device() == "cuda"
    print("Parte 1: CUDA disponibile -> 'cuda' (ha priorita' anche se anche MPS risultasse disponibile) — OK")


def part2_mps_used_when_cuda_unavailable():
    _install_fake_torch(cuda_available=False, mps_available=True)
    from common.device import detect_default_device
    assert detect_default_device() == "mps"
    print("Parte 2: CUDA assente, MPS disponibile -> 'mps' (caso Mac Apple Silicon) — OK")


def part3_cpu_when_neither_available():
    _install_fake_torch(cuda_available=False, mps_available=False)
    from common.device import detect_default_device
    assert detect_default_device() == "cpu"
    print("Parte 3: ne' CUDA ne' MPS disponibili -> 'cpu' (fallback finale) — OK")


def part4_missing_mps_backend_attribute_falls_back_to_cpu_not_crash():
    """Su una versione di torch senza `torch.backends.mps` (pre-1.12 circa)
    la funzione non deve sollevare AttributeError -- deve trattarlo come
    "MPS non disponibile" e ripiegare su cpu (o cuda se disponibile)."""
    _install_fake_torch(cuda_available=False, mps_available=None)
    from common.device import detect_default_device
    assert detect_default_device() == "cpu"
    print("Parte 4: un torch senza torch.backends.mps (versioni vecchie) non fa esplodere la "
          "funzione, ripiega su 'cpu' invece di sollevare AttributeError — OK")


def main():
    part1_cuda_available_wins_over_mps()
    part2_mps_used_when_cuda_unavailable()
    part3_cpu_when_neither_available()
    part4_missing_mps_backend_attribute_falls_back_to_cpu_not_crash()
    print("\nVerifica completata senza errori: detect_default_device() sceglie CUDA > MPS > CPU "
          "in ogni combinazione, senza richiedere torch/una GPU vera in questo ambiente.")


if __name__ == "__main__":
    main()
