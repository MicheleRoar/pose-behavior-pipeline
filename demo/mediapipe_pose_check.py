"""
mediapipe_pose_check.py
=========================
Verifica della logica pura di `pose/mediapipe_pose.py` (rimappatura
BlazePose -> COCO-17, calcolo del box di ritaglio con padding) SENZA
mediapipe/camera installati: usa oggetti landmark sintetici (x, y,
visibility normalizzati, come restituiti da MediaPipe) invece di una vera
inferenza.

Esegui con: python mediapipe_pose_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.keypoints import KP
from pose.mediapipe_pose import BLAZEPOSE_TO_COCO, blazepose_to_coco, padded_crop_box, _empty_pose


class _FakeLandmark:
    """Sostituto minimo di un NormalizedLandmark di MediaPipe: solo i campi
    letti da blazepose_to_coco (x, y normalizzati 0-1, visibility)."""
    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def make_landmarks(overrides: dict[int, tuple[float, float, float]]) -> list[_FakeLandmark]:
    """33 landmark BlazePose, di default al centro (0.5, 0.5) con
    visibility 0.0 (cosi' i test possono verificare che gli indici NON
    sovrascritti restino "vuoti" nell'output), con gli indici in
    `overrides` impostati esplicitamente a (x, y, visibility)."""
    landmarks = [_FakeLandmark(0.5, 0.5, 0.0) for _ in range(33)]
    for idx, (x, y, vis) in overrides.items():
        landmarks[idx] = _FakeLandmark(x, y, vis)
    return landmarks


def part1_all_coco_joints_get_mapped():
    """Ogni voce di BLAZEPOSE_TO_COCO deve tradursi nel giunto COCO giusto,
    con le coordinate normalizzate riportate correttamente in pixel del
    frame intero (offset + scala del ritaglio)."""
    overrides = {blaze_idx: (0.5, 0.5, 0.9) for blaze_idx in BLAZEPOSE_TO_COCO}
    landmarks = make_landmarks(overrides)
    # ritaglio di 100x200 px con angolo in alto a sinistra a (300, 50) nel
    # frame intero: il centro normalizzato (0.5, 0.5) deve finire a
    # (300 + 50, 50 + 100) = (350, 150) in pixel del frame intero.
    kxy, kconf = blazepose_to_coco(landmarks, frame_offset_xy=(300.0, 50.0), crop_size_wh=(100.0, 200.0))

    for coco_name in BLAZEPOSE_TO_COCO.values():
        idx = KP[coco_name]
        assert np.allclose(kxy[idx], [350.0, 150.0]), (
            f"{coco_name}: atteso (350, 150), trovato {kxy[idx]}"
        )
        assert abs(kconf[idx] - 0.9) < 1e-6, f"{coco_name}: attesa confidenza 0.9, trovata {kconf[idx]}"
    print(f"Parte 1: tutti i {len(BLAZEPOSE_TO_COCO)} giunti COCO mappati da BlazePose finiscono "
          "nelle coordinate pixel attese (offset + scala del ritaglio applicati correttamente) — OK")


def part2_per_joint_confidence_is_preserved_not_averaged():
    """Ogni giunto COCO deve riportare la visibility del SUO landmark
    BlazePose specifico, non un valore medio/condiviso -- un giunto
    incerto (bassa visibility) non deve "contaminare" la confidenza di un
    giunto ben rilevato nello stesso frame."""
    landmarks = make_landmarks({
        0: (0.5, 0.5, 0.95),   # nose: alta confidenza
        7: (0.5, 0.5, 0.10),   # left_ear: bassa confidenza
    })
    kxy, kconf = blazepose_to_coco(landmarks, frame_offset_xy=(0.0, 0.0), crop_size_wh=(100.0, 100.0))

    assert abs(kconf[KP["nose"]] - 0.95) < 1e-6, f"nose: attesa confidenza 0.95, trovata {kconf[KP['nose']]}"
    assert abs(kconf[KP["left_ear"]] - 0.10) < 1e-6, (
        f"left_ear: attesa confidenza 0.10, trovata {kconf[KP['left_ear']]}"
    )
    print("Parte 2: la confidenza per giunto riflette la visibility del SUO landmark BlazePose "
          "specifico (nose=0.95, left_ear=0.10 restano distinti, non mescolati) — OK")


def part2b_empty_pose_is_all_nan_zero_confidence():
    """_empty_pose() (usata quando nessuna posa e' rilevata nel ritaglio)
    deve avere la stessa forma (17, 2)/(17,) di un rilevamento vero, tutta
    NaN/confidenza 0 -- cosi' il resto della pipeline (che si aspetta
    sempre un array di quella forma) non deve gestire un caso speciale."""
    kxy, kconf = _empty_pose()
    assert kxy.shape == (17, 2) and kconf.shape == (17,), f"forma inattesa: {kxy.shape}, {kconf.shape}"
    assert np.isnan(kxy).all(), "kxy di _empty_pose() deve essere tutto NaN"
    assert (kconf == 0.0).all(), "kconf di _empty_pose() deve essere tutto zero"
    print("Parte 2b: _empty_pose() ha la forma corretta (17,2)/(17,), tutta NaN/confidenza 0 — OK")


def part3_padded_crop_box_clamped_to_frame():
    """Il box di ritaglio va allargato del padding richiesto ma MAI oltre i
    bordi del frame (altrimenti si tenterebbe di ritagliare pixel
    inesistenti)."""
    frame_shape = (480, 640)  # (h, w)

    # box centrale: il padding non deve toccare i bordi
    bbox = np.array([200.0, 150.0, 300.0, 350.0])
    x1, y1, x2, y2 = padded_crop_box(bbox, frame_shape, padding=0.1)
    bw, bh = 100.0, 200.0
    assert x1 == int(200 - bw * 0.1) and y1 == int(150 - bh * 0.1)
    assert x2 == int(300 + bw * 0.1) and y2 == int(350 + bh * 0.1)

    # box vicino al bordo in alto a sinistra: il padding sfonderebbe (x<0,
    # y<0) senza il clamping
    bbox_edge = np.array([5.0, 5.0, 60.0, 80.0])
    x1e, y1e, x2e, y2e = padded_crop_box(bbox_edge, frame_shape, padding=0.5)
    assert x1e == 0 and y1e == 0, f"atteso clamping a (0,0), trovato ({x1e},{y1e})"

    # box vicino al bordo in basso a destra: stesso discorso su x2/y2
    bbox_edge2 = np.array([600.0, 440.0, 638.0, 478.0])
    x1e2, y1e2, x2e2, y2e2 = padded_crop_box(bbox_edge2, frame_shape, padding=0.5)
    assert x2e2 == 640 and y2e2 == 480, f"atteso clamping a (640,480), trovato ({x2e2},{y2e2})"

    print("Parte 3: il box di ritaglio con padding resta sempre dentro i bordi del frame — OK")


def part3b_resolve_model_path_existing_explicit_path_used_as_is():
    """Un path passato ESPLICITAMENTE dall'utente e che esiste gia' -- vale
    per qualunque nome di file, anche il default nudo se per caso esiste
    nella cwd corrente -- va usato cosi' com'e', nessuna risoluzione/
    download (nato dal bug reale: 'unable to find pose_landmarker_lite'
    quando si lanciava da una cwd diversa da quella del curl manuale, vedi
    il docstring di mediapipe_pose.py)."""
    import shutil
    import tempfile
    from pose.mediapipe_pose import _resolve_model_path

    tmp_dir = tempfile.mkdtemp()
    try:
        custom_path = str(Path(tmp_dir) / "pose_landmarker_full.task")
        Path(custom_path).write_bytes(b"fake model bytes")
        result = _resolve_model_path(custom_path)
        assert result == custom_path, f"un path esistente va restituito invariato, ottenuto {result}"
        print("Parte 3b: _resolve_model_path() restituisce invariato un path esplicito gia' "
              "esistente, nessuna risoluzione/download tentati — OK")
    finally:
        shutil.rmtree(tmp_dir)


def part3c_resolve_model_path_custom_missing_path_not_touched():
    """Path custom (nome diverso dal default nudo) che NON esiste: nessun
    tentativo di indovinare/scaricare -- restituito invariato, cosi' il
    chiamante (MediaPipe) fallisce con l'errore originale, piu' chiaro di
    un download silenzioso nel posto sbagliato."""
    from pose.mediapipe_pose import _resolve_model_path

    missing = "/percorso/inesistente/pose_landmarker_full.task"
    result = _resolve_model_path(missing)
    assert result == missing
    print("Parte 3c: _resolve_model_path() non tocca un path custom mancante (nome diverso dal "
          "default nudo) — OK")


def part3d_resolve_model_path_bare_default_uses_project_cache():
    """Il nome nudo di default ('pose_landmarker_lite.task', quello
    risolto in modo cwd-dipendente prima di questo fix) viene risolto
    nella cache fissa del progetto -- riusa il file se gia' presente
    (nessun download ripetuto), lo scarica (qui: un finto che crea solo un
    file vuoto, nessuna rete vera) se assente."""
    import os
    import shutil
    import tempfile
    import pose.mediapipe_pose as mp_pose

    tmp_dir = tempfile.mkdtemp()
    original_cache = mp_pose._DEFAULT_MODEL_CACHE_PATH
    original_download = mp_pose._download_pose_landmarker_lite
    original_cwd = os.getcwd()
    # _resolve_model_path() controlla PRIMA se il nome nudo esiste gia'
    # nella cwd corrente (comportamento voluto: un file gia' scaricato a
    # mano li' va rispettato) -- ci si sposta in una cartella pulita per
    # non dipendere da eventuali file lasciati da run precedenti (es. una
    # vecchia src/pose_landmarker_lite.task scaricata prima di questo fix).
    os.chdir(tmp_dir)
    try:
        # caso 1: gia' in cache -- nessun download
        cache_path = Path(tmp_dir) / "models" / "pose_landmarker_lite.task"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(b"already downloaded")
        mp_pose._DEFAULT_MODEL_CACHE_PATH = cache_path
        download_calls = []
        mp_pose._download_pose_landmarker_lite = lambda dest: download_calls.append(dest)

        result = mp_pose._resolve_model_path("pose_landmarker_lite.task")
        assert result == str(cache_path)
        assert download_calls == [], "il file e' gia' nella cache -- nessun download da rifare"

        # caso 2: assente -- il download (finto) viene invocato e crea il file
        cache_path.unlink()

        def fake_download(dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"downloaded now")

        mp_pose._download_pose_landmarker_lite = fake_download
        result2 = mp_pose._resolve_model_path("pose_landmarker_lite.task")
        assert result2 == str(cache_path)
        assert cache_path.exists(), "il download finto avrebbe dovuto creare il file"

        print("Parte 3d: _resolve_model_path() risolve il nome nudo di default nella cache fissa "
              "del progetto -- riusa il file se presente, lo scarica se assente — OK")
    finally:
        mp_pose._DEFAULT_MODEL_CACHE_PATH = original_cache
        mp_pose._download_pose_landmarker_lite = original_download
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir)


def _find_pose_model() -> str | None:
    """Percorso del modello pose_landmarker_lite.task se presente in uno
    dei posti noti -- la nuova cache fissa del progetto (models/ alla
    root, vedi _resolve_model_path in mediapipe_pose.py) o la vecchia
    convenzione accanto a src/ (per compatibilita' con chi l'aveva gia'
    scaricato a mano prima di questo fix). None se assente ovunque, cosi'
    i test che richiedono mediapipe/il modello vero si saltano da soli con
    una nota invece di far fallire l'intera suite su una macchina senza il
    modello scaricato."""
    from pose.mediapipe_pose import _DEFAULT_MODEL_CACHE_PATH
    candidates = [
        _DEFAULT_MODEL_CACHE_PATH,
        Path(__file__).resolve().parent.parent / "src" / "pose_landmarker_lite.task",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def part4_defensive_clamp_absorbs_duplicate_or_out_of_order_timestamps():
    """`MediaPipeCropPoseEstimator.estimate()` include una clamp difensiva:
    anche chiamandolo due volte con LO STESSO timestamp (il bug segnalato
    dall'utente: due persone nello stesso frame sulla stessa istanza) o con
    un timestamp addirittura MINORE del precedente, non deve sollevare
    'Input timestamp must be monotonically increasing' -- il valore
    effettivo viene silenziosamente forzato a crescere. Questa e' una rete
    di sicurezza aggiuntiva, NON il fix principale: il fix principale e'
    non riusare mai la stessa istanza per persone diverse, vedi
    MediaPipePoseByTrack (parte 5)."""
    import mediapipe  # noqa: F401 -- solo per il check di disponibilita'
    model_path = _find_pose_model()
    if model_path is None:
        print("Parte 4: SALTATA (pose_landmarker_lite.task non trovato accanto a src/ in questo "
              "ambiente) — va verificata sul Mac dove il modello e' scaricato.")
        return

    from pose.mediapipe_pose import MediaPipeCropPoseEstimator

    est = MediaPipeCropPoseEstimator(model_path=model_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox = np.array([20.0, 20.0, 150.0, 200.0])
    est.estimate(frame, bbox, timestamp_ms=1000)
    est.estimate(frame, bbox, timestamp_ms=1000)  # stesso timestamp di nuovo: non deve sollevare
    est.estimate(frame, bbox, timestamp_ms=500)   # timestamp addirittura MINORE: non deve sollevare
    print("Parte 4: la clamp difensiva in MediaPipeCropPoseEstimator.estimate() assorbe timestamp "
          "duplicati/fuori ordine (lo scenario esatto del bug segnalato) senza sollevare "
          "'monotonically increasing' — OK")


def part5_pool_per_track_gives_each_person_an_independent_stream():
    """MediaPipePoseByTrack (il fix principale): due persone nello stesso
    frame (stesso timestamp), passate con due track_id diversi, ottengono
    ciascuna la propria istanza `MediaPipeCropPoseEstimator` indipendente
    (oggetti Python distinti, non solo "nessun crash" -- vedi il docstring
    del modulo sul perche' condividerne una manderebbe in crash E
    mescolerebbe lo smussamento temporale di persone diverse), su piu'
    frame consecutivi, e forget() rimuove solo il track indicato."""
    import mediapipe  # noqa: F401
    model_path = _find_pose_model()
    if model_path is None:
        print("Parte 5: SALTATA (pose_landmarker_lite.task non trovato) — va verificata sul Mac.")
        return

    from pose.mediapipe_pose import MediaPipePoseByTrack

    pool = MediaPipePoseByTrack(model_path=model_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox_a = np.array([10.0, 10.0, 100.0, 150.0])
    bbox_b = np.array([150.0, 10.0, 260.0, 150.0])

    for frame_idx in range(3):  # alcuni frame "consecutivi" per entrambe le persone
        now_ms = int((frame_idx / 15.0) * 1000)
        kxy_a, kconf_a = pool.estimate(track_id=1, frame_bgr=frame, bbox=bbox_a, timestamp_ms=now_ms)
        kxy_b, kconf_b = pool.estimate(track_id=2, frame_bgr=frame, bbox=bbox_b, timestamp_ms=now_ms)
        assert kxy_a.shape == (17, 2) and kconf_a.shape == (17,)
        assert kxy_b.shape == (17, 2) and kconf_b.shape == (17,)

    assert set(pool._estimators.keys()) == {1, 2}, "atteso un'istanza per ciascun track_id visto"
    assert pool._estimators[1] is not pool._estimators[2], (
        "le due persone devono avere istanze DISTINTE, non la stessa condivisa"
    )
    pool.forget(1)
    assert set(pool._estimators.keys()) == {2}, "forget() deve rimuovere solo l'istanza del track indicato"
    print("Parte 5: MediaPipePoseByTrack da' a ciascun track_id la propria istanza/il proprio "
          "'stream' indipendente (due oggetti distinti, non condivisi) su piu' frame consecutivi, "
          "senza crash — OK (fix del bug segnalato)")


def main():
    part1_all_coco_joints_get_mapped()
    part2_per_joint_confidence_is_preserved_not_averaged()
    part2b_empty_pose_is_all_nan_zero_confidence()
    part3_padded_crop_box_clamped_to_frame()
    part3b_resolve_model_path_existing_explicit_path_used_as_is()
    part3c_resolve_model_path_custom_missing_path_not_touched()
    part3d_resolve_model_path_bare_default_uses_project_cache()
    part4_defensive_clamp_absorbs_duplicate_or_out_of_order_timestamps()
    part5_pool_per_track_gives_each_person_an_independent_stream()
    print("\nVerifica completata senza errori: la rimappatura BlazePose -> COCO-17, il calcolo "
          "del box di ritaglio, e (dove mediapipe/il modello sono disponibili) il fix del bug "
          "'monotonically increasing' in mediapipe_pose.py si comportano come atteso.")


if __name__ == "__main__":
    main()
