"""
video_player_check.py
=======================
Verifica della logica di cache/seek in `gui/video_player.py` SENZA GUI, video
reale o tracker (niente Tkinter, niente ultralytics): inietta un generatore
sintetico che produce `RunnerFrame` numerati, cosi' si puo' controllare
esattamente quale frame viene restituito da ogni Avanti/Indietro e quante
volte il generatore viene effettivamente avanzato (per dimostrare che
"Indietro" non fa MAI nuova inferenza).

Esegui con: python video_player_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from gui.pipeline_runner import RunnerFrame
from gui.video_player import VideoPlayer


def make_generator_factory(n_frames: int, counter: dict):
    """Fabbrica di generatori sintetici: ogni frame e' un piccolo array con
    un solo pixel che codifica il proprio indice, cosi' i test possono
    verificare "quale frame e' questo" senza un video vero. `counter["calls"]`
    conta quante volte la fabbrica e' stata invocata (quante volte
    VideoPlayer e' "ripartito da zero") e `counter["frames_produced"]` quanti
    frame sono stati effettivamente generati (prova indiretta che la cache
    evita di rigenerarli)."""
    def factory():
        counter["calls"] = counter.get("calls", 0) + 1
        def gen():
            for i in range(n_frames):
                counter["frames_produced"] = counter.get("frames_produced", 0) + 1
                frame = np.full((2, 2, 3), i, dtype=np.uint8)
                yield RunnerFrame(frame=frame, rows=[{"frame_idx": i}], now=float(i), mode="test")
        return gen()
    return factory


def frame_index_of(runner_frame: RunnerFrame) -> int:
    return int(runner_frame.frame[0, 0, 0])


def part1_forward_advances_and_produces_new_frames():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))

    seen = [frame_index_of(player.step_forward()) for _ in range(5)]
    assert seen == [0, 1, 2, 3, 4], f"atteso [0..4] in ordine, trovato {seen}"
    assert counter["frames_produced"] == 5, "atteso esattamente 5 frame generati (uno per step_forward)"
    # l'esaurimento si scopre solo tentando di andare OLTRE l'ultimo frame
    # (il generatore stesso non sa di essere all'ultimo elemento finche' non
    # gli si chiede il successivo) -- prima di questa chiamata is_exhausted
    # deve quindi essere ancora False.
    assert not player.is_exhausted, "non deve segnalarsi esaurito prima di aver tentato oltre l'ultimo frame"
    assert player.step_forward() is None, "oltre la fine deve restituire None, non sollevare o ripartire"
    assert player.is_exhausted, "dopo il tentativo oltre l'ultimo frame il player deve segnalarsi esaurito"
    print("Parte 1: 5x step_forward() restituisce i frame 0..4 nell'ordine giusto, "
          "esattamente 5 frame generati, fine-video rilevata correttamente — OK")


def part2_back_never_reprocesses():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))

    for _ in range(3):
        player.step_forward()  # cursore a 0,1,2 -> in cache i frame 0,1,2
    assert counter["frames_produced"] == 3

    back1 = frame_index_of(player.step_back())
    back2 = frame_index_of(player.step_back())
    assert (back1, back2) == (1, 0), f"atteso (1,0) tornando indietro da 2, trovato {(back1, back2)}"
    assert counter["frames_produced"] == 3, (
        "step_back() non deve MAI generare nuovi frame (deve leggere solo dalla cache): "
        f"attesi ancora 3 frame prodotti, trovati {counter['frames_produced']}"
    )
    assert player.step_back() is None, "al frame 0 non si puo' tornare ulteriormente indietro"
    print("Parte 2: step_back() rilegge dalla cache (0 nuovi frame generati) e si ferma "
          "correttamente al primo frame — OK")


def part3_forward_after_back_resumes_from_cache_then_live():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))

    for _ in range(4):
        player.step_forward()  # cache: 0,1,2,3 -- cursore a 3
    player.step_back()          # cursore a 2 (rilegge dalla cache)
    player.step_back()          # cursore a 1 (rilegge dalla cache)
    assert counter["frames_produced"] == 4

    # da qui in poi, "avanti" deve rileggere dalla cache (2, poi 3) PRIMA di
    # tornare a generare frame nuovi (4) -- non deve rigenerare 2 e 3.
    resumed = [frame_index_of(player.step_forward()) for _ in range(3)]
    assert resumed == [2, 3, 4], f"atteso [2,3,4] riprendendo da cursore=1, trovato {resumed}"
    assert counter["frames_produced"] == 5, (
        "solo il frame 4 (mai visto prima) doveva generare nuova inferenza: "
        f"attesi 5 frame prodotti in totale, trovati {counter['frames_produced']}"
    )
    print("Parte 3: dopo essere tornato indietro, avanti rilegge prima dalla cache (2,3) "
          "poi riprende l'elaborazione live solo per il frame davvero nuovo (4) — OK")


def part4_reset_starts_a_fresh_generator():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))
    player.step_forward()
    player.step_forward()
    assert counter["calls"] == 1

    player.reset()
    assert player.current is None, "dopo reset() non deve esserci alcun frame 'in vista'"
    assert player.cached_frame_count == 0, "dopo reset() la cache deve essere vuota"

    first_after_reset = frame_index_of(player.step_forward())
    assert first_after_reset == 0, "dopo reset() si deve ripartire dal frame 0 di un NUOVO generatore"
    assert counter["calls"] == 2, "reset() deve invocare di nuovo la generator_factory (nuova sessione)"
    print("Parte 4: reset() svuota la cache e riparte da un generatore nuovo di zecca — OK")


def part5_all_rows_matches_cached_frames_in_order():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(4, counter))
    for _ in range(4):
        player.step_forward()

    rows = player.all_rows()
    assert [r["frame_idx"] for r in rows] == [0, 1, 2, 3], (
        f"atteso una riga per frame nell'ordine 0..3, trovato {[r['frame_idx'] for r in rows]}"
    )
    print("Parte 5: all_rows() concatena le righe di ogni frame in cache, nell'ordine giusto — OK")


def part6_seek_jumps_within_cache_without_reprocessing():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))
    for _ in range(4):
        player.step_forward()  # cache: 0,1,2,3 -- cursore a 3
    assert counter["frames_produced"] == 4

    jumped = frame_index_of(player.seek(1))
    assert jumped == 1, f"atteso di saltare al frame 1, trovato {jumped}"
    assert player.cursor == 1
    assert counter["frames_produced"] == 4, "seek() dentro la cache non deve generare nuovi frame"

    back_to_edge = frame_index_of(player.seek(3))
    assert back_to_edge == 3
    assert counter["frames_produced"] == 4

    # fuori dalla cache (frame 4 non ancora elaborato): seek() non deve
    # spostare il cursore ne' generare nulla -- e' compito del chiamante
    # usare step_forward() per l'elaborazione di recupero.
    assert player.seek(4) is None
    assert player.cursor == 3, "un seek() oltre la cache non deve spostare il cursore"
    assert counter["frames_produced"] == 4
    print("Parte 6: seek() salta istantaneamente dentro la cache gia' elaborata, senza mai "
          "generare nuovi frame, e rifiuta un salto oltre il prefisso gia' elaborato — OK")


def main():
    part1_forward_advances_and_produces_new_frames()
    part2_back_never_reprocesses()
    part3_forward_after_back_resumes_from_cache_then_live()
    part4_reset_starts_a_fresh_generator()
    part5_all_rows_matches_cached_frames_in_order()
    part6_seek_jumps_within_cache_without_reprocessing()
    print("\nVerifica completata senza errori: VideoPlayer avanza/generera' nuovi frame solo "
          "quando serve davvero, non rigenera mai un frame gia' visto tornando indietro, e "
          "reset() riparte pulito da un nuovo generatore.")


if __name__ == "__main__":
    main()
