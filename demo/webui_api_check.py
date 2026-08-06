"""
webui_api_check.py
====================
Verifica della logica PURA di `webui/api.py` -- niente pywebview, niente
finestra vera, niente video/tracker reale (stesso spirito di
video_player_check.py per gui/video_player.py). Copre le quattro funzioni/
classi isolate apposta per essere testabili senza una finestra:
`build_player_kwargs`, `encode_frame_jpeg_b64`, `_LatencyTracker`,
`build_status`. Non tocca la classe `Api` stessa (quella richiede
pywebview/una finestra vera -- verificata a mano sul Mac, come le altre
feature della GUI).

Esegui con: python webui_api_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from gui.pipeline_runner import RunnerFrame
from webui.api import (
    build_player_kwargs, encode_frame_jpeg_b64, _LatencyTracker, build_status,
    probe_video_metadata,
)


def part1_build_player_kwargs_mirrors_app_py_defaults():
    kwargs = build_player_kwargs({
        "mode": "pose", "source": "video.mp4", "fps": "15",
        "with_hands": True, "with_eyes": True,
    })
    assert kwargs["mode"] == "pose"
    assert kwargs["source"] == "video.mp4"
    assert kwargs["fps"] == 15.0
    # device non specificato -> None, NON "mps": la risoluzione automatica
    # (cuda/mps/cpu) avviene in Api.build_player(), non qui, cosi' questa
    # funzione resta pura/testabile senza richiedere torch installato --
    # vedi anche part1b sotto per il passthrough di un device esplicito.
    assert kwargs["device"] is None
    assert kwargs["pose_model"] == "yolo26s-pose.pt"  # scale default "s"
    assert kwargs["seg_model"] == "yolo26s-seg.pt"
    assert kwargs["with_hands"] is True
    assert kwargs["with_eyes"] is True
    assert kwargs["with_mouth"] is False
    assert kwargs["max_people"] is None
    assert kwargs["with_reid"] is False  # nessun max_people -> reid ignorata anche se richiesta
    print("Parte 1: build_player_kwargs applica i default giusti e passa i flag richiesti — OK")


def part1b_explicit_device_passes_through_unchanged():
    """Se JS manda esplicitamente un device (es. l'utente vuole forzare
    "cpu" per debug), build_player_kwargs deve limitarsi a passarlo cosi'
    com'e' -- l'auto-rilevamento in Api.build_player() scatta SOLO quando
    il campo e' assente/vuoto, non deve mai scavalcare una scelta esplicita."""
    kwargs = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": 15, "device": "cuda",
    })
    assert kwargs["device"] == "cuda"
    print("Parte 1b: un device esplicito nei parametri passa invariato, senza auto-rilevamento — OK")


def part2_hands_face_ignored_outside_pose_and_both():
    # Stessa regola di app.py::_on_mode_change: mani/viso valgono solo in
    # Pose/Both, in Segmentation vengono azzerati anche se il chiamante li
    # manda a True per errore.
    kwargs = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": 15,
        "with_hands": True, "with_eyes": True, "with_mouth": True,
    })
    assert kwargs["with_hands"] is False
    assert kwargs["with_eyes"] is False
    assert kwargs["with_mouth"] is False
    print("Parte 2: mani/viso vengono ignorati fuori da Pose/Both — OK")


def part3_mediapipe_pose_only_in_segmentation():
    kwargs_seg = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": 15,
        "with_mediapipe_pose": True,
    })
    assert kwargs_seg["with_mediapipe_pose"] is True

    kwargs_pose = build_player_kwargs({
        "mode": "pose", "source": "v.mp4", "fps": 15,
        "with_mediapipe_pose": True,
    })
    assert kwargs_pose["with_mediapipe_pose"] is False
    print("Parte 3: MediaPipe pose-per-maschera attivabile solo in modalita' segmentation — OK")


def part4_reid_requires_max_people_and_right_mode():
    # reid richiesta ma senza max_people -> ignorata (non solleva errore)
    kwargs = build_player_kwargs({
        "mode": "both", "source": "v.mp4", "fps": 15, "reid": True,
    })
    assert kwargs["with_reid"] is False
    assert kwargs["with_seg_reid"] is False

    # reid richiesta con max_people, modalita' "both" -> entrambe le reid attive
    kwargs2 = build_player_kwargs({
        "mode": "both", "source": "v.mp4", "fps": 15, "reid": True, "max_people": "3",
    })
    assert kwargs2["max_people"] == 3
    assert kwargs2["with_reid"] is True
    assert kwargs2["with_seg_reid"] is True

    # reid richiesta con max_people, modalita' "pose" -> solo pose reid attiva
    kwargs3 = build_player_kwargs({
        "mode": "pose", "source": "v.mp4", "fps": 15, "reid": True, "max_people": 2,
    })
    assert kwargs3["with_reid"] is True
    assert kwargs3["with_seg_reid"] is False
    print("Parte 4: re-id/seg-reid attive solo con max_people impostato e nella modalita' giusta — OK")


def part5_invalid_mode_and_missing_source_raise():
    try:
        build_player_kwargs({"mode": "bogus", "source": "v.mp4", "fps": 15})
        raise AssertionError("doveva sollevare ValueError per mode sconosciuto")
    except ValueError:
        pass
    try:
        build_player_kwargs({"mode": "pose", "source": "", "fps": 15})
        raise AssertionError("doveva sollevare ValueError per source mancante")
    except ValueError:
        pass
    print("Parte 5: mode sconosciuto o source mancante sollevano ValueError (build_player le trasforma "
          "in {'ok': False, 'error': ...} invece di far esplodere la chiamata JS) — OK")


def part6_encode_frame_jpeg_b64_roundtrip_and_resize():
    import base64
    import cv2

    small = np.zeros((10, 10, 3), dtype=np.uint8)
    small[:] = (0, 128, 255)  # BGR
    data_url = encode_frame_jpeg_b64(small, max_width=1600)
    assert data_url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (10, 10, 3)  # nessun resize sotto max_width

    wide = np.zeros((100, 3200, 3), dtype=np.uint8)
    data_url2 = encode_frame_jpeg_b64(wide, max_width=1600)
    raw2 = base64.b64decode(data_url2.split(",", 1)[1])
    decoded2 = cv2.imdecode(np.frombuffer(raw2, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded2.shape[1] == 1600  # ridimensionato verso il basso
    assert decoded2.shape[0] == 50    # proporzioni mantenute (100 * 1600/3200)
    print("Parte 6: encode_frame_jpeg_b64 produce un data-URL decodificabile e ridimensiona solo "
          "verso il basso oltre max_width — OK")


def part7_latency_tracker_rolling_average():
    tracker = _LatencyTracker(window=3)
    assert tracker.avg_latency_ms == 0.0
    assert tracker.processing_fps == 0.0

    tracker.record(0.100)  # 100ms
    tracker.record(0.100)
    assert abs(tracker.avg_latency_ms - 100.0) < 1e-6
    assert abs(tracker.processing_fps - 10.0) < 1e-6

    # la finestra e' di 3: un quarto valore fa uscire il piu' vecchio
    tracker.record(0.100)
    tracker.record(0.400)  # ora la finestra contiene [0.1, 0.1, 0.4] -> media 0.2
    assert abs(tracker.avg_latency_ms - 200.0) < 1e-6
    print("Parte 7: _LatencyTracker calcola una media mobile reale, non un valore finto — OK")


def part8_build_status_uses_people_count_not_len_rows():
    frame = RunnerFrame(frame=np.zeros((2, 2, 3), dtype=np.uint8), rows=[], now=1.5,
                         mode="pose", people_count=4)
    latency = _LatencyTracker()
    latency.record(0.050)
    status = build_status(runner_frame=frame, cached_frame_count=7, latency=latency,
                           device="mps", mode="pose", is_finished=False)
    assert status["people_count"] == 4  # da RunnerFrame.people_count, non da len(rows)=0
    assert status["rows_this_frame"] == 0
    assert status["frame_index"] == 6
    assert status["timecode_s"] == 1.5
    assert status["device"] == "mps"
    assert status["is_finished"] is False
    assert abs(status["avg_latency_ms"] - 50.0) < 1e-6
    print("Parte 8: build_status legge people_count da RunnerFrame (affidabile anche a rows vuote) — OK")


def part9_probe_video_metadata_missing_file_returns_none_not_zero():
    meta = probe_video_metadata("/tmp/definitely_not_a_real_video_file_xyz.mp4")
    assert meta == {"frame_count": None, "duration_s": None, "container_fps": None}, (
        "un file inesistente/illeggibile deve dare 'sconosciuto' (None), non 0 -- "
        "0 farebbe credere a un video vuoto invece che a una durata non calcolabile"
    )
    print("Parte 9: probe_video_metadata su un file inesistente ritorna 'sconosciuto' (None), non zero — OK")


def part10_probe_video_metadata_reads_real_container_metadata():
    import tempfile
    import os
    import cv2

    path = tempfile.mktemp(suffix=".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (16, 16))
    for _ in range(30):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()
    try:
        meta = probe_video_metadata(path)
        assert meta["frame_count"] == 30, f"attesi 30 frame nel container, trovato {meta['frame_count']}"
        assert abs(meta["container_fps"] - 10.0) < 1e-6
        assert abs(meta["duration_s"] - 3.0) < 1e-6  # 30 frame / 10 fps = 3s
    finally:
        os.remove(path)
    print("Parte 10: probe_video_metadata legge SOLO i metadati del container (frame count/fps/durata) "
          "da un file video vero, senza decodificare i frame uno per uno — OK")


def part11_build_status_carries_totals_and_max_people_for_the_timeline():
    frame = RunnerFrame(frame=np.zeros((2, 2, 3), dtype=np.uint8), rows=[], now=6.3,
                         mode="segmentation", people_count=2)
    latency = _LatencyTracker()
    status = build_status(runner_frame=frame, cached_frame_count=209, latency=latency,
                           device="mps", mode="segmentation", is_finished=False,
                           max_people=20, total_frame_count=1087, total_duration_s=72.8)
    assert status["frame_index"] == 208
    assert status["total_frame_count"] == 1087  # per il timecode/metrica "corrente / totale"
    assert status["total_duration_s"] == 72.8
    assert status["max_people"] == 20  # per la metrica "tracce attive: 2 / 20"
    print("Parte 11: build_status porta anche i totali (frame/durata) e max_people, per il timecode "
          "'corrente / totale' e la metrica 'tracce attive: N / max' del nuovo layout — OK")


def part12_seg_backend_defaults_to_yolo_and_passes_through():
    kwargs_default = build_player_kwargs({"mode": "segmentation", "source": "v.mp4", "fps": "15"})
    assert kwargs_default["seg_backend"] == "yolo"
    assert kwargs_default["sam_chunk_size"] == 600
    assert kwargs_default["sam_overlap"] == 50
    assert kwargs_default["sam_chunk_store_dir"] is None

    kwargs_sam = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": "15",
        "seg_backend": "sam31", "sam_chunk_size": "300", "sam_overlap": "20",
        "sam_chunk_store_dir": "/tmp/chunks",
    })
    assert kwargs_sam["seg_backend"] == "sam31"
    assert kwargs_sam["sam_chunk_size"] == 300  # stringa -> int, come max_people altrove
    assert kwargs_sam["sam_overlap"] == 20
    assert kwargs_sam["sam_chunk_store_dir"] == "/tmp/chunks"
    print("Parte 12: seg_backend/sam_chunk_size/sam_overlap/sam_chunk_store_dir hanno i default giusti "
          "('yolo'/600/50/None) e passano attraverso invariati quando specificati — OK")


def part13_sam_redetect_every_and_text_prompt_defaults_and_passthrough():
    kwargs_default = build_player_kwargs({"mode": "segmentation", "source": "v.mp4", "fps": "15"})
    assert kwargs_default["sam_redetect_every"] is None
    assert kwargs_default["sam_text_prompt"] is None

    kwargs_set = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": "15",
        "seg_backend": "sam31", "sam_redetect_every": "120", "sam_text_prompt": "person",
    })
    assert kwargs_set["sam_redetect_every"] == 120  # stringa -> int
    assert kwargs_set["sam_text_prompt"] == "person"

    # stringa vuota == non impostato, come max_people altrove (non "0"/"" letterale)
    kwargs_empty = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": "15",
        "sam_redetect_every": "", "sam_text_prompt": "",
    })
    assert kwargs_empty["sam_redetect_every"] is None
    assert kwargs_empty["sam_text_prompt"] is None
    print("Parte 13: sam_redetect_every/sam_text_prompt hanno default None e passano attraverso "
          "invariati quando specificati (stringa vuota == non impostato) — OK")


if __name__ == "__main__":
    part1_build_player_kwargs_mirrors_app_py_defaults()
    part1b_explicit_device_passes_through_unchanged()
    part2_hands_face_ignored_outside_pose_and_both()
    part3_mediapipe_pose_only_in_segmentation()
    part4_reid_requires_max_people_and_right_mode()
    part5_invalid_mode_and_missing_source_raise()
    part6_encode_frame_jpeg_b64_roundtrip_and_resize()
    part7_latency_tracker_rolling_average()
    part8_build_status_uses_people_count_not_len_rows()
    part9_probe_video_metadata_missing_file_returns_none_not_zero()
    part10_probe_video_metadata_reads_real_container_metadata()
    part12_seg_backend_defaults_to_yolo_and_passes_through()
    part13_sam_redetect_every_and_text_prompt_defaults_and_passthrough()
    part11_build_status_carries_totals_and_max_people_for_the_timeline()
    part12_seg_backend_defaults_to_yolo_and_passes_through()
    print("\nVerifica completata senza errori: la logica pura di webui/api.py (parametri, codifica "
          "frame, metriche, metadati video) si comporta come atteso, senza bisogno di pywebview o di "
          "una finestra vera.")
