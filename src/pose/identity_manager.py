"""
identity_manager.py
====================
Livello di decisione condiviso per "Identity & Re-identification", usato sia
dalla pipeline pose (`reid.py`, firma antropometrica) sia da quella di
segmentazione (`seg_reid.py`, centroide/colore/forma della maschera). Questo
modulo NON ricalcola nessun segnale di somiglianza -- quelli restano
interamente in `reid.py`/`seg_reid.py` (proporzioni corporee, colore,
posizione, silhouette...), con tutte le loro regole di causalita' e i
guardrail gia' documentati li'. Quello che centralizza qui:

  1. `IdentityMode` / `SessionMode`: il vocabolario condiviso esposto dalla
     sezione "Identity & Re-identification" della GUI (vedi webui/app.js),
     invece di due superfici di parametri che rischiano di divergere.

  2. Assegnazione batch, globalmente ottima (algoritmo ungherese, via
     `scipy.optimize.linear_sum_assignment`): quando PIU' persone "in
     attesa di verifica" diventano pronte nello STESSO frame, la versione
     precedente (loop sequenziale in `reid.py`/`seg_reid.py`) le risolveva
     una alla volta nell'ordine di iterazione, mutando la mappa delle
     persone "perse" man mano -- corretto (non permette un doppio aggancio
     nello stesso frame) ma non necessariamente la coppia migliore in
     assoluto quando ci sono piu' candidati e piu' identita' perse in
     competizione. Qui si costruisce una matrice di costo candidati x
     identita' perse e si risolve in un colpo solo.

  3. Tre esiti invece di un binario match/non-match: `matched` (punteggio
     comodamente sopra soglia) / `new` (nessun candidato plausibile,
     conia una nuova identita') / `uncertain` (zona grigia tra le due
     soglie -- per la policy "meglio frammentare un'identita' che
     scambiarne due in silenzio", un candidato incerto viene SEGNALATO ma
     NON fuso automaticamente).

Convenzione: questo modulo lavora sempre in spazio di COSTO (valori piu'
bassi = piu' simili), coerente con la distanza RMS gia' usata da reid.py.
`seg_reid.py`, che internamente usa un punteggio di similarita' 0..1 (piu'
alto = piu' simile), lo converte con `cost = 1 - score` prima di chiamare
`resolve_batch()` -- vedi li' per il wiring. Le coppie impossibili per la
regola di causalita' (una persona "persa" DOPO che il candidato era gia'
apparso) vanno passate come `inf`, mai omesse: `resolve_batch()` le tratta
come "quella coppia non esiste" senza escludere la riga/colonna
dall'assegnazione ungherese (che richiede una matrice completa).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class IdentityMode(str, Enum):
    """Cosa deve fare la pipeline con gli id, dal meno al piu' persistente.
    Interpretato dal chiamante (pipeline_runner.py / webui/api.py): questo
    modulo non decide da solo se istanziare un ReIdentifier/SegReIdentifier,
    si limita a nominare le tre opzioni in un unico posto."""

    FRAME_BY_FRAME = "frame_by_frame"   # nessun id persistente tra un frame e l'altro
    TRACKING_ONLY = "tracking_only"     # id mantenuto finche' la traccia del tracker e' continua
    TRACKING_REID = "tracking_reid"     # + recupero dopo uscita/occlusione/perdita del track


class SessionMode(str, Enum):
    """Quante persone ci si aspetta nella sessione. Non aggiunge una nuova
    meccanica: si traduce nei parametri gia' esistenti nei due motori
    (`max_people=1` per SINGLE lato reid.py/seg_reid.py, tetto piu' alto o
    assente per MULTIPLE) -- vedi `suggested_max_people_policy()` sotto."""

    SINGLE = "single"
    MULTIPLE = "multiple"


def wants_reid_engine(mode: IdentityMode) -> bool:
    """True solo se la modalita' richiede di istanziare ReIdentifier/
    SegReIdentifier (TRACKING_REID). FRAME_BY_FRAME e TRACKING_ONLY usano
    entrambe direttamente gli id grezzi del tracker sottostante (ByteTrack/
    SAM) -- la differenza tra le due e' se quell'id viene anche esposto come
    "identita' stabile" nel CSV/overlay (TRACKING_ONLY) o se ogni frame e'
    trattato come indipendente anche a parita' di track_id (FRAME_BY_FRAME,
    utile solo per confronti "senza alcuna assunzione di continuita'")."""
    return mode == IdentityMode.TRACKING_REID


@dataclass
class IdentityManagerConfig:
    """Configurazione condivisa esposta dalla sezione "Identity &
    Re-identification" della GUI."""

    mode: IdentityMode = IdentityMode.TRACKING_REID
    session_mode: SessionMode = SessionMode.MULTIPLE
    max_people: int | None = None
    lost_identity_memory_s: float = 180.0
    # policy di matching: sempre conservativa per costruzione (vedi
    # `resolve_batch`) -- questo flag controlla SOLO se un candidato in
    # "zona grigia" viene segnalato come `uncertain` (default) o trattato
    # come `new` senza traccia (comportamento precedente, binario).
    flag_uncertain: bool = True
    # Sotto (in costo, cioe' <=) = fusione automatica. Sopra `reject_cost`
    # (escluso) = nuova identita', nessun candidato plausibile. Nel mezzo =
    # zona grigia ("uncertain").
    accept_cost: float = 0.12
    reject_cost: float = 0.20


@dataclass
class AssignmentOutcome:
    candidate_index: int              # riga nella matrice di costo passata a resolve_batch
    matched_lost_id: int | None       # identita' (person_id) rivendicata, se presente
    cost: float | None
    status: str                       # "matched" | "new" | "uncertain"


def resolve_batch(cost_matrix: np.ndarray, lost_ids: list[int],
                   config: IdentityManagerConfig) -> list[AssignmentOutcome]:
    """Risolve congiuntamente quale candidato "in attesa di verifica" (righe)
    rivendica quale identita' persa (colonne), dato una matrice di costo
    (valori piu' bassi = piu' simili; `inf`/`nan` per le coppie impossibili
    per causalita' o segnale insufficiente). Un esito per ogni riga, sempre
    -- anche con `lost_ids` vuoto (tutti "new") o con piu' candidati che
    identita' perse (i candidati in eccesso, non assegnabili dall'algoritmo
    ungherese su una matrice rettangolare, restano "new").

    Risolto con `scipy.optimize.linear_sum_assignment` (minimizza il costo
    totale): quando piu' persone rientrano nello stesso frame e competono
    per le stesse identita' perse, questo trova l'accoppiamento
    complessivamente migliore invece di quello che il primo candidato
    processato "si accaparra" per primo (loop sequenziale precedente).
    """
    n_candidates = cost_matrix.shape[0]
    if n_candidates == 0:
        return []
    if not lost_ids:
        return [AssignmentOutcome(i, None, None, "new") for i in range(n_candidates)]

    from scipy.optimize import linear_sum_assignment

    # Le coppie impossibili (inf/nan) diventano un costo enorme ma finito:
    # linear_sum_assignment richiede valori finiti. Abbastanza grande da non
    # essere mai scelto a meno che sia l'UNICA opzione rimasta per quella
    # riga/colonna (in tal caso viene comunque scartato subito dopo dal
    # controllo reject_cost, vedi sotto).
    safe_cost = np.where(np.isfinite(cost_matrix), cost_matrix, 1e6)
    row_ind, col_ind = linear_sum_assignment(safe_cost)
    assigned_col = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    outcomes: list[AssignmentOutcome] = []
    for i in range(n_candidates):
        col = assigned_col.get(i)
        cost = float(cost_matrix[i, col]) if col is not None else None
        if col is None or cost is None or not np.isfinite(cost) or cost > config.reject_cost:
            outcomes.append(AssignmentOutcome(i, None, cost, "new"))
        elif cost <= config.accept_cost:
            outcomes.append(AssignmentOutcome(i, int(lost_ids[col]), cost, "matched"))
        else:
            # zona grigia: l'algoritmo ungherese propone comunque questa
            # coppia come la migliore disponibile, ma non abbastanza sicura
            # per fondere in automatico -- vedi il docstring del modulo.
            if config.flag_uncertain:
                outcomes.append(AssignmentOutcome(i, int(lost_ids[col]), cost, "uncertain"))
            else:
                outcomes.append(AssignmentOutcome(i, None, cost, "new"))
    return outcomes


def suggested_max_people_policy(session_mode: SessionMode, configured_max_people: int | None) -> int | None:
    """Traduce `SessionMode` nel tetto rigido `max_people` gia' capito da
    ReIdentifier/SegReIdentifier. SINGLE forza 1 (una sola persona attesa,
    il recupero dell'identita' puo' essere il piu' permissivo possibile: chi
    rientra e' per forza quella persona -- vedi `_force_match_at_capacity` in
    reid.py / il vincolo assoluto in seg_reid.py). MULTIPLE rispetta il
    valore configurato dall'utente (puo' restare `None`: nessun tetto, mai
    forzare un aggancio solo per "non poter essere una persona in piu'")."""
    if session_mode == SessionMode.SINGLE:
        return 1
    return configured_max_people
