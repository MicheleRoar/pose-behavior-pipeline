"""
appearance_embedding_check.py
===============================
Verifica delle funzioni pure di `pose/appearance_embedding.py`
(`embedding_similarity`, `ema_update`, `torchreid_available`) SENZA
richiedere 'torch'/'torchreid' installati -- `OSNetEmbedder` stesso (che
li richiede) NON e' testata qui, per costruzione: il suo comportamento e'
verificato solo strutturalmente (revisione del codice, formato dei crop
atteso da `torchreid.utils.FeatureExtractor`) e per via indiretta nei test
di integrazione di `reid_check.py`/`seg_reid_check.py`, che usano uno stub
(`_FakeEmbedder`) con la stessa interfaccia invece del vero OSNet.

Esegui con: python appearance_embedding_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.appearance_embedding import embedding_similarity, ema_update, torchreid_available


def embedding_similarity_matches_cosine_convention():
    identical = np.array([1.0, 0.0, 0.0])
    assert np.isclose(embedding_similarity(identical, identical), 1.0), "stesso vettore -> similarita' 1.0"

    orthogonal_a, orthogonal_b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.isclose(embedding_similarity(orthogonal_a, orthogonal_b), 0.5), (
        "vettori ortogonali (coseno 0) -> similarita' 0.5 nella convenzione 0..1 del modulo"
    )

    opposite_a, opposite_b = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    assert np.isclose(embedding_similarity(opposite_a, opposite_b), 0.0), (
        "vettori opposti (coseno -1) -> similarita' 0.0"
    )

    assert embedding_similarity(None, identical) is None, "un embedding assente -> None, mai un valore inventato"
    assert embedding_similarity(identical, None) is None
    print("embedding_similarity: convenzione coseno [-1,1] -> [0,1] rispettata, None propagato correttamente — OK")


def ema_update_converges_and_stays_normalized():
    rng = np.random.default_rng(5)
    true_direction = np.array([1.0, 0.0, 0.0])
    ema = None
    for _ in range(50):
        # osservazioni rumorose attorno alla direzione vera, poi rinormalizzate
        # (come farebbe un vero embedding L2-normalizzato di OSNet per frame)
        noisy = true_direction + rng.normal(0, 0.3, size=3)
        noisy = noisy / np.linalg.norm(noisy)
        ema = ema_update(ema, noisy, alpha=0.9)
        assert np.isclose(np.linalg.norm(ema), 1.0, atol=1e-6), "l'EMA deve restare un vettore unitario"

    sim = embedding_similarity(ema, true_direction)
    assert sim > 0.95, (
        f"dopo 50 aggiornamenti rumorosi, l'EMA deve essersi stabilizzata vicino alla direzione "
        f"vera (similarita'={sim:.3f}) -- e' esattamente il comportamento richiesto ('restare in "
        f"memoria e affinarsi nel tempo', l'idea di StrongSORT citata nel docstring del modulo)"
    )
    print(f"ema_update: dopo 50 osservazioni rumorose, similarita' con la direzione vera={sim:.3f} "
          "(converge, resta unitaria) — OK")


def ema_update_handles_missing_values():
    v = np.array([1.0, 0.0])
    assert ema_update(None, v) is v, "nessuna storia precedente -> l'osservazione diventa la memoria"
    assert ema_update(v, None) is v, "nessuna nuova osservazione -> la memoria precedente resta invariata"
    assert ema_update(None, None) is None
    print("ema_update: casi limite (nessuna storia / nessuna osservazione / nessuno dei due) — OK")


def torchreid_available_is_a_safe_boolean_check():
    # Non asseriamo ne' True ne' False (dipende dalla macchina): solo che la
    # funzione non solleva mai un errore e restituisce un booleano vero e
    # proprio -- e' pensata per il gating della checkbox in webui/app.js,
    # deve essere sicura da chiamare anche senza 'torch' installato.
    result = torchreid_available()
    assert isinstance(result, bool)
    print(f"torchreid_available(): {result} (nessun errore sollevato, indipendentemente dall'esito) — OK")


def main():
    embedding_similarity_matches_cosine_convention()
    ema_update_converges_and_stays_normalized()
    ema_update_handles_missing_values()
    torchreid_available_is_a_safe_boolean_check()
    print("\nVerifica completata senza errori: le funzioni pure di appearance_embedding.py "
          "si comportano come atteso, senza richiedere 'torch'/'torchreid' installati.")


if __name__ == "__main__":
    main()
