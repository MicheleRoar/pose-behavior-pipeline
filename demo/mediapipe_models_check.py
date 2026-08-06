"""
mediapipe_models_check.py
============================
Verifica `common/mediapipe_models.py::resolve_model_path()` -- la logica
condivisa di risoluzione/download automatico dei modelli MediaPipe Tasks
usata da `pose/mediapipe_pose.py`, `pose/hands.py` e `pose/gaze_head.py`
(estratta da li' per non triplicarla) -- SENZA importare `mediapipe` ne'
fare download di rete veri (`_download` sostituita da un finto che crea
solo un file vuoto).

Nato da un bug reale osservato da Michele su due modelli diversi
(pose_landmarker_lite.task, poi hand_landmarker.task/face_landmarker.task
con lo stesso identico problema): lanciando la pipeline da una cwd
diversa da quella in cui aveva scaricato il file a mano, MediaPipe
falliva con "unable to find <nome modello>" -- causa: il default era il
nome nudo del file, risolto da MediaPipe come path RELATIVO ALLA CWD.

Uso:
    python demo/mediapipe_models_check.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import common.mediapipe_models as mm  # noqa: E402

_FAKE_URL = "https://example.invalid/models/fake_landmarker/float16/1/fake_landmarker.task"
_FAKE_BASENAME = "fake_landmarker.task"


def part1_existing_explicit_path_used_as_is():
    # Un path che l'utente ha passato ESPLICITAMENTE e che esiste gia' --
    # nessuna risoluzione/download, usato cosi' com'e' anche se il nome
    # non ha nulla a che vedere con quello di default.
    tmp_dir = tempfile.mkdtemp()
    try:
        custom_path = os.path.join(tmp_dir, "un_nome_qualsiasi.task")
        Path(custom_path).write_bytes(b"fake model bytes")
        result = mm.resolve_model_path(custom_path, download_url=_FAKE_URL)
        assert result == custom_path, f"un path esistente va restituito invariato, ottenuto {result}"
        print("PASS part1_existing_explicit_path_used_as_is")
    finally:
        shutil.rmtree(tmp_dir)


def part2_custom_missing_path_not_touched():
    # Path custom (nome diverso dal default nudo) che NON esiste: nessun
    # tentativo di indovinare/scaricare -- restituito invariato, cosi' il
    # chiamante (MediaPipe) fallisce con l'errore originale, piu' chiaro
    # di un download silenzioso nel posto sbagliato.
    missing = "/percorso/inesistente/un_nome_qualsiasi.task"
    result = mm.resolve_model_path(missing, download_url=_FAKE_URL)
    assert result == missing
    print("PASS part2_custom_missing_path_not_touched")


def part3_bare_default_name_reuses_existing_cache_file():
    tmp_dir = tempfile.mkdtemp()
    original_cache_dir = mm.MODELS_CACHE_DIR
    original_download = mm._download
    original_cwd = os.getcwd()
    # resolve_model_path() controlla PRIMA se il nome nudo esiste gia'
    # nella cwd corrente -- ci si sposta in una cartella pulita per non
    # dipendere da eventuali file lasciati da run precedenti.
    os.chdir(tmp_dir)
    download_calls = []
    mm._download = lambda url, dest: download_calls.append((url, dest))
    try:
        cache_dir = Path(tmp_dir) / "cache"
        cache_path = cache_dir / _FAKE_BASENAME
        cache_dir.mkdir()
        cache_path.write_bytes(b"already downloaded")
        mm.MODELS_CACHE_DIR = cache_dir

        result = mm.resolve_model_path(_FAKE_BASENAME, download_url=_FAKE_URL)
        assert result == str(cache_path)
        assert download_calls == [], "il file e' gia' nella cache -- nessun download da rifare"
        print("PASS part3_bare_default_name_reuses_existing_cache_file")
    finally:
        mm.MODELS_CACHE_DIR = original_cache_dir
        mm._download = original_download
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir)


def part4_bare_default_name_triggers_download_when_missing():
    tmp_dir = tempfile.mkdtemp()
    original_cache_dir = mm.MODELS_CACHE_DIR
    original_download = mm._download
    original_cwd = os.getcwd()
    os.chdir(tmp_dir)

    def fake_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded now")

    mm._download = fake_download
    try:
        cache_dir = Path(tmp_dir) / "cache"
        cache_path = cache_dir / _FAKE_BASENAME
        mm.MODELS_CACHE_DIR = cache_dir
        assert not cache_path.exists(), "precondizione: il file non deve esistere ancora"

        result = mm.resolve_model_path(_FAKE_BASENAME, download_url=_FAKE_URL)
        assert result == str(cache_path)
        assert cache_path.exists(), "il download finto avrebbe dovuto creare il file"
        print("PASS part4_bare_default_name_triggers_download_when_missing")
    finally:
        mm.MODELS_CACHE_DIR = original_cache_dir
        mm._download = original_download
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir)


def part5_default_basename_derived_from_url_not_hardcoded():
    # Il nome "di default" non e' una costante fissa: e' derivato
    # dall'ultimo pezzo di download_url -- cosi' lo stesso helper serve
    # per pose_landmarker_lite.task, hand_landmarker.task E
    # face_landmarker.task senza bisogno di parametrizzare il nome a parte.
    other_url = "https://example.invalid/models/altro/float16/1/altro_landmarker.task"
    missing_altro = "altro_landmarker.task"  # nudo, ma per un URL diverso da _FAKE_URL
    result = mm.resolve_model_path(missing_altro, download_url=_FAKE_URL)
    # il basename di missing_altro ("altro_landmarker.task") NON combacia col
    # basename di _FAKE_URL ("fake_landmarker.task") -- trattato come path
    # custom mancante, non risolto in cache.
    assert result == missing_altro, (
        "un nome nudo che non combacia col basename di download_url va trattato come path "
        "custom (non risolto/scaricato), non solo perche' 'sembra' un modello MediaPipe"
    )
    del other_url  # solo per chiarezza del commento sopra, non usato oltre
    print("PASS part5_default_basename_derived_from_url_not_hardcoded")


def main():
    part1_existing_explicit_path_used_as_is()
    part2_custom_missing_path_not_touched()
    part3_bare_default_name_reuses_existing_cache_file()
    part4_bare_default_name_triggers_download_when_missing()
    part5_default_basename_derived_from_url_not_hardcoded()
    print("\nTutti i test di common/mediapipe_models.py (resolve_model_path, senza mediapipe/rete) "
          "sono passati.")


if __name__ == "__main__":
    main()
