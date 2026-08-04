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

    tracker = PoseTracker(model_name="yolo26n-pose.pt", device="mps")
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
    model_name : nome/percorso del modello (es. "yolo26n-pose.pt" per il
        modello nano, più veloce; "yolo26s-pose.pt"/"yolo26m-pose.pt" per
        maggiore accuratezza a scapito della velocità — in batch offline,
        senza vincolo di tempo reale, conviene usare il modello più grande
        che il device riesce a sostenere).
    device : "mps" su Apple Silicon, "cpu" come fallback, "cuda" se
        disponibile una GPU NVIDIA.
    conf_threshold : soglia di confidenza minima per considerare valida una
        detection. Di default 0.1 (non 0.4): ByteTrack ha una fase di
        recupero a bassa confidenza (track_low_thresh, 0.1 di default in
        bytetrack.yaml) pensata apposta per gestire detection deboli senza
        perdere l'identità — un conf_threshold troppo alto le scarta prima
        che ByteTrack possa usarle, causando ID spuri su scene difficili
        (visione dall'alto, movimento rapido, illuminazione artificiale).
    tracker : config di tracking Ultralytics ("bytetrack.yaml" di default,
        oppure "bytetrack_permissive.yaml" per scene con cali di confidenza
        frequenti e non dovuti a vera occlusione — vedi quel file per i
        dettagli sui parametri).
    max_people : se impostato, limita il numero di persone per frame a
        questo valore, tenendo solo le detection con confidenza più alta
        (utile quando si conosce a priori il numero di partecipanti alla
        sessione, es. 2 per 1v1 bambino-caregiver, o una decina per una
        sessione di gruppo: sopprime detection spurie da rumore/riflessi/
        doppie-detection sopra quel numero prima che diventino un track).
        Non risolve il problema di una persona reale che perde e riprende
        un ID dopo una vera occlusione — per quello vedi reid.py e
        bytetrack_permissive.yaml. None (default) = nessun limite.
    """

    def __init__(self, model_name: str = "yolo26n-pose.pt", device: str = "mps",
                 conf_threshold: float = 0.1, tracker: str = "bytetrack.yaml",
                 max_people: int | None = None):
        # Import ritardato: così il resto del pacchetto (features.py,
        # anonymize.py) resta utilizzabile/testabile anche senza
        # ultralytics/torch installati (utile per test unitari leggeri).
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = conf_threshold
        self.tracker = tracker
        self.max_people = max_people

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
                box_conf = r.boxes.conf.cpu().numpy()        # (n_people,) confidenza detection
                track_ids = r.boxes.id.cpu().numpy().astype(int)

                order = range(len(track_ids))
                if self.max_people is not None and len(track_ids) > self.max_people:
                    # Tiene solo le `max_people` detection più sicure di
                    # questo frame: sopprime il rumore in eccesso senza
                    # mai scartare persone reali quando il numero rientra
                    # nel limite (il filtro non scatta sotto la soglia).
                    order = np.argsort(-box_conf)[: self.max_people]

                for idx in order:
                    people.append((int(track_ids[idx]), kpts_xy[idx], kpts_conf[idx]))
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
