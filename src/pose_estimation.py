"""
pose_estimation.py
===================
Wrapper sottile su Ultralytics YOLO-pose per estrarre keypoint COCO-17
multi-persona con tracking, da un file video o da una sorgente live
(es. webcam / Canon R8 via EOS Webcam Utility, o capture card HDMI).

Pensato per girare su Apple Silicon (M1/M2/...) sfruttando il backend MPS
(`device="mps"`); su altre macchine cade automaticamente su CPU/CUDA se
disponibile.

Questo modulo richiede `ultralytics` e `opencv-python`, non installati
nell'ambiente sandbox usato per sviluppare/testare `features.py`
(vedi README per le istruzioni di installazione locale sul Mac).

Esempio d'uso:

    from pose_estimation import PoseTracker

    tracker = PoseTracker(model_name="yolov8n-pose.pt", device="mps")
    for result in tracker.run(source=0):   # 0 = prima webcam disponibile
        for track_id, kpts, conf in result.people:
            ...  # kpts: array (17, 2), conf: array (17,)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameResult:
    frame_index: int
    frame: np.ndarray
    people: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    # ogni elemento: (track_id, keypoints (17,2), confidences (17,))


class PoseTracker:
    """Estrae keypoint multi-persona con tracking da una sorgente video
    usando Ultralytics YOLO-pose.

    Parameters
    ----------
    model_name : nome/percorso del modello (es. "yolov8n-pose.pt" per il
        modello nano, più veloce; "yolov8s-pose.pt" per maggiore accuratezza
        a scapito della velocità).
    device : "mps" su Apple Silicon, "cpu" come fallback, "cuda" se
        disponibile una GPU NVIDIA.
    conf_threshold : soglia di confidenza minima per considerare valida una
        detection.
    tracker : algoritmo di tracking Ultralytics ("bytetrack.yaml" di default).
    """

    def __init__(self, model_name: str = "yolov8n-pose.pt", device: str = "mps",
                 conf_threshold: float = 0.4, tracker: str = "bytetrack.yaml"):
        # Import ritardato: così il resto del pacchetto (features.py,
        # anonymize.py) resta utilizzabile/testabile anche senza
        # ultralytics/torch installati (utile per test unitari leggeri).
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = conf_threshold
        self.tracker = tracker

    def run(self, source, stream: bool = True):
        """Esegue la pose estimation + tracking sulla sorgente indicata.

        `source` può essere:
        - un intero (indice webcam, es. 0)
        - il percorso di un file video
        - una stringa di stream (es. RTSP)

        Restituisce un generatore di `FrameResult`.
        """
        results = self.model.track(
            source=source,
            device=self.device,
            conf=self.conf_threshold,
            tracker=self.tracker,
            stream=stream,
            verbose=False,
        )

        for i, r in enumerate(results):
            people = []
            if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
                kpts_xy = r.keypoints.xy.cpu().numpy()       # (n_people, 17, 2)
                kpts_conf = r.keypoints.conf.cpu().numpy()   # (n_people, 17)
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                for tid, kxy, kconf in zip(track_ids, kpts_xy, kpts_conf):
                    people.append((int(tid), kxy, kconf))
            yield FrameResult(frame_index=i, frame=r.orig_img, people=people)


def keypoints_dict_to_array(kpts_xy: np.ndarray) -> np.ndarray:
    """Utility per garantire la shape (17, 2) attesa da features.py, anche
    se il modello restituisce un numero diverso di keypoint (es. modelli
    custom): qui si assume schema COCO-17 standard di Ultralytics.
    """
    assert kpts_xy.shape[-2:] == (17, 2), (
        f"Attesi 17 keypoint (schema COCO), ricevuti shape {kpts_xy.shape}. "
        "Se usi un modello custom, aggiorna keypoints.py di conseguenza."
    )
    return kpts_xy
