"""
video_player.py
=================
Stato del "player" usato da app.py: avanti/indietro sui frame gia'
elaborati, ripresa dell'elaborazione live quando si supera l'ultimo frame in
cache.

Perche' una cache invece di un "seek" vero: i tracker (ByteTrack, e
opzionalmente reid.py/seg_reid.py) sono intrinsecamente sequenziali e
stateful -- non si puo' saltare a un frame arbitrario nel mezzo di un video
senza rielaborare tutto da capo, altrimenti si perderebbe la storia su cui
si basano re-id e finestre scorrevoli delle feature. Quindi:
- "Indietro" rilegge dalla cache gia' calcolata: istantaneo, nessuna nuova
  inferenza;
- "Avanti" oltre l'ultimo frame in cache riprende l'elaborazione live dal
  generatore `iter_pipeline_frames()` (vedi pipeline_runner.py), aggiungendo
  ogni nuovo frame alla cache mano a mano che viene prodotto.

`VideoPlayer` non sa nulla di Tkinter/OpenCV/disegno: riceve un
`generator_factory` (di norma `lambda: iter_pipeline_frames(**kwargs)`) e
restituisce oggetti `RunnerFrame` gia' pronti (overlay disegnato incluso).
Questo lo rende testabile senza video reale, vedi
`demo/video_player_check.py`, che inietta un generatore sintetico.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional

from gui.pipeline_runner import RunnerFrame


class VideoPlayer:
    def __init__(self, generator_factory: Callable[[], Iterator[RunnerFrame]]):
        """`generator_factory` viene richiamato per creare un NUOVO
        generatore ogni volta che serve ripartire da capo (prima chiamata a
        step_forward()/step_back(), o dopo reset()) -- non un generatore gia'
        istanziato, perche' un generatore esaurito non si puo' "riavvolgere"."""
        self._generator_factory = generator_factory
        self._generator: Optional[Iterator[RunnerFrame]] = None
        self._cache: list[RunnerFrame] = []
        self._cursor: int = -1  # indice nella cache del frame attualmente mostrato
        self._exhausted: bool = False  # il generatore ha esaurito i frame (fine video)

    @property
    def current(self) -> Optional[RunnerFrame]:
        """Il frame attualmente "in vista" (l'ultimo restituito da
        step_forward()/step_back()), o None se non e' ancora stato mostrato
        alcun frame."""
        if 0 <= self._cursor < len(self._cache):
            return self._cache[self._cursor]
        return None

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def cached_frame_count(self) -> int:
        return len(self._cache)

    @property
    def at_live_edge(self) -> bool:
        """True se il cursore e' sull'ultimo frame gia' elaborato: il
        prossimo step_forward() richiedera' nuova inferenza, non solo la
        cache."""
        return self._cursor == len(self._cache) - 1

    @property
    def is_exhausted(self) -> bool:
        """True se il video e' finito (il generatore ha smesso di produrre
        frame) E siamo arrivati fino in fondo alla cache."""
        return self._exhausted and self.at_live_edge

    def step_forward(self) -> Optional[RunnerFrame]:
        """Avanza di un frame: dalla cache se gia' disponibile (istantaneo),
        altrimenti elabora il prossimo frame dal vivo. Ritorna None se il
        video e' gia' del tutto esaurito."""
        if self._cursor + 1 < len(self._cache):
            self._cursor += 1
            return self._cache[self._cursor]

        if self._exhausted:
            return None

        if self._generator is None:
            self._generator = self._generator_factory()

        try:
            frame = next(self._generator)
        except StopIteration:
            self._exhausted = True
            return None

        self._cache.append(frame)
        self._cursor += 1
        return frame

    def step_back(self) -> Optional[RunnerFrame]:
        """Torna indietro di un frame, sempre dalla cache. Ritorna None se
        si e' gia' al primo frame (o se non e' ancora stato mostrato
        nulla)."""
        if self._cursor <= 0:
            return None
        self._cursor -= 1
        return self._cache[self._cursor]

    def seek(self, index: int) -> Optional[RunnerFrame]:
        """Salta ISTANTANEAMENTE a `index` DENTRO la cache gia' elaborata
        (nessuna nuova inferenza) -- usato dallo scrubber della timeline
        della GUI web, che per lo stesso motivo spiegato nel docstring del
        modulo puo' offrire un seek libero solo sul prefisso gia' elaborato.
        Ritorna None (senza toccare il cursore) se `index` e' fuori dalla
        cache: a differenza di step_forward(), questo metodo non estende MAI
        la cache -- chi chiama deve usare step_forward() in sequenza per
        raggiungere un punto non ancora elaborato (elaborazione di
        recupero), non questo metodo."""
        if 0 <= index < len(self._cache):
            self._cursor = index
            return self._cache[self._cursor]
        return None

    def all_rows(self) -> list[dict]:
        """Tutte le righe dati (una per persona per frame) accumulate finora
        nella cache, nell'ordine dei frame -- usato per il CSV finale."""
        rows: list[dict] = []
        for f in self._cache:
            rows.extend(f.rows)
        return rows

    def reset(self) -> None:
        """Ricomincia da capo: nuova sessione di inferenza, cache svuotata.
        Necessario ogni volta che cambiano i parametri (modello, feature,
        --max-people, ecc.), perche' un tracker gia' avviato non si puo'
        riconfigurare a meta' strada -- vedi il docstring del modulo."""
        self._generator = None
        self._cache = []
        self._cursor = -1
        self._exhausted = False
