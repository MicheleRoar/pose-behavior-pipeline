"""
features.py
============
Estrazione di feature comportamentali da sequenze temporali di keypoint 2D
(schema COCO-17), pensate come "mattoncini" riutilizzabili per l'analisi di
video di interazione bambino-caregiver.

Design:
- Nessuna dipendenza da un modello di pose estimation specifico: tutte le
  funzioni lavorano su array numpy (n_frame, 17, 2/3), quindi funzionano sia
  con output di Ultralytics YOLO-pose sia con MediaPipe/OpenPose, una volta
  rimappati sullo schema COCO-17.
- Le feature sono scelte per riflettere marker discussi in letteratura per
  lo studio del neurosviluppo infantile via video (vedi README):
    * angoli articolari e loro variabilità (postura, controllo motorio)
    * velocità/energia di movimento
    * indice di simmetria sinistra/destra
    * "score" di movimento ripetitivo (stereotipie) via autocorrelazione
    * prossimità e sincronia tra due persone tracciate (bambino/caregiver)

Nota metodologica: queste feature sono un punto di partenza esplorativo, non
marker diagnostici validati. Qualunque uso clinico richiede validazione su
dati annotati e supervisione di personale qualificato.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from keypoints import KP, LR_PAIRS, JOINT_ANGLE_TRIPLETS
from geometry import angle_at as _angle_at


def compute_joint_angles(frame_kpts: np.ndarray) -> dict[str, float]:
    """Calcola gli angoli articolari definiti in JOINT_ANGLE_TRIPLETS per un
    singolo frame.

    Parameters
    ----------
    frame_kpts : array (17, 2) con le coordinate (x, y) dei keypoint COCO-17.
    """
    angles = {}
    for name, (a_name, b_name, c_name) in JOINT_ANGLE_TRIPLETS.items():
        a, b, c = frame_kpts[KP[a_name]], frame_kpts[KP[b_name]], frame_kpts[KP[c_name]]
        angles[name] = _angle_at(a, b, c)
    return angles


# ---------------------------------------------------------------------------
# Cinematica: velocità ed energia di movimento
# ---------------------------------------------------------------------------

def compute_joint_speed(kpts_sequence: np.ndarray, fps: float) -> pd.DataFrame:
    """Velocità (unità/secondo) di ciascun keypoint lungo la sequenza.

    Parameters
    ----------
    kpts_sequence : array (n_frame, 17, 2)
    fps : frame rate del video

    Returns
    -------
    DataFrame (n_frame, 17) con la velocità istantanea di ogni keypoint
    (il primo frame è NaN, non essendoci un frame precedente).
    """
    diffs = np.diff(kpts_sequence, axis=0) * fps  # unità/secondo
    speed = np.linalg.norm(diffs, axis=2)  # (n_frame-1, 17)
    speed = np.vstack([np.full((1, speed.shape[1]), np.nan), speed])
    return pd.DataFrame(speed, columns=[k for k in KP])


def movement_energy(kpts_sequence: np.ndarray, fps: float) -> np.ndarray:
    """Proxy dell'energia cinetica complessiva per frame: somma delle
    velocità al quadrato su tutti i keypoint. Utile come indice sintetico
    di "quanto si muove" una persona in un dato istante.
    """
    speed_df = compute_joint_speed(kpts_sequence, fps)
    return (speed_df ** 2).sum(axis=1).to_numpy()


# ---------------------------------------------------------------------------
# Simmetria sinistra/destra
# ---------------------------------------------------------------------------

def symmetry_index(kpts_sequence: np.ndarray, fps: float) -> pd.Series:
    """Indice di simmetria per ciascuna coppia sinistra/destra, definito come

        SI = |v_left - v_right| / (v_left + v_right)

    calcolato sulla velocità media di ciascun arto lungo la sequenza.
    SI = 0 -> perfettamente simmetrico, SI -> 1 -> fortemente asimmetrico.
    Riferimento concettuale: studi su asimmetrie di movimento e midline
    postural control in ASD (vedi README).
    """
    speed_df = compute_joint_speed(kpts_sequence, fps)
    out = {}
    for left, right in LR_PAIRS:
        v_l = speed_df[left].mean(skipna=True)
        v_r = speed_df[right].mean(skipna=True)
        denom = v_l + v_r
        out[f"{left.replace('left_', '')}_symmetry"] = (
            abs(v_l - v_r) / denom if denom > 1e-8 else np.nan
        )
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Movimento ripetitivo / stereotipie
# ---------------------------------------------------------------------------

def repetitive_motion_score(signal: np.ndarray, fps: float,
                             min_freq_hz: float = 0.5,
                             max_freq_hz: float = 8.0) -> dict[str, float]:
    """Quantifica quanto un segnale 1D (es. velocità di un polso) è
    dominato da un'oscillazione periodica in una banda di frequenza
    plausibile per movimenti ripetitivi manuali (stereotipie).

    Approccio: densità spettrale di potenza (FFT) + rapporto tra la potenza
    del picco dominante nella banda [min_freq_hz, max_freq_hz] e la potenza
    totale del segnale ("peak power ratio"). Un valore alto indica un
    movimento fortemente periodico in quella banda.

    Nota: la banda di default è un punto di partenza ragionevole per
    movimenti manuali ripetitivi; la banda ottimale va validata su dati
    reali/annotati per il contesto clinico specifico.
    """
    signal = np.nan_to_num(signal - np.nanmean(signal))
    n = len(signal)
    if n < 8:
        return {"peak_freq_hz": np.nan, "peak_power_ratio": np.nan}

    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal)) ** 2

    band_mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    total_power = power[1:].sum()  # esclude componente DC
    if total_power < 1e-12 or not band_mask.any():
        return {"peak_freq_hz": np.nan, "peak_power_ratio": 0.0}

    band_power = power[band_mask]
    band_freqs = freqs[band_mask]
    peak_idx = np.argmax(band_power)

    return {
        "peak_freq_hz": float(band_freqs[peak_idx]),
        "peak_power_ratio": float(band_power[peak_idx] / total_power),
    }


# ---------------------------------------------------------------------------
# Interazione tra due persone (es. bambino-caregiver)
# ---------------------------------------------------------------------------

def hip_center(frame_kpts: np.ndarray) -> np.ndarray:
    """Centro del bacino, usato come proxy della posizione del corpo."""
    return (frame_kpts[KP["left_hip"]] + frame_kpts[KP["right_hip"]]) / 2.0


def proximity_series(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Distanza euclidea tra i centri-bacino di due persone tracciate,
    frame per frame. Le due sequenze devono avere la stessa lunghezza ed
    essere allineate temporalmente (stesso indice frame).
    """
    centers_a = np.array([hip_center(f) for f in seq_a])
    centers_b = np.array([hip_center(f) for f in seq_b])
    return np.linalg.norm(centers_a - centers_b, axis=1)


def windowed_synchrony(signal_a: np.ndarray, signal_b: np.ndarray,
                        window: int, step: int) -> pd.DataFrame:
    """Correlazione di Pearson tra due segnali di movimento (es. energia
    cinetica di bambino e caregiver) calcolata su finestre scorrevoli, come
    proxy semplice di sincronia motoria diadica.
    """
    n = min(len(signal_a), len(signal_b))
    rows = []
    for start in range(0, n - window + 1, step):
        a = signal_a[start:start + window]
        b = signal_b[start:start + window]
        if np.nanstd(a) < 1e-8 or np.nanstd(b) < 1e-8:
            corr = np.nan
        else:
            corr = float(np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))[0, 1])
        rows.append({"frame_start": start, "frame_end": start + window, "synchrony": corr})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Postura, attività e auto-contatto
# ---------------------------------------------------------------------------

def torso_length(frame_kpts: np.ndarray) -> float:
    """Distanza spalle-bacino in un frame, usata come unità di scala della
    persona (invariante rispetto alla distanza dalla camera) per normalizzare
    altre feature (escursione verticale, self-touch)."""
    shoulder_center = (frame_kpts[KP["left_shoulder"]] + frame_kpts[KP["right_shoulder"]]) / 2.0
    return float(np.linalg.norm(shoulder_center - hip_center(frame_kpts)))


def vertical_excursion(kpts_sequence: np.ndarray, normalize: bool = True) -> float:
    """Escursione verticale (max - min) del centro-bacino lungo la sequenza:
    proxy di transizioni posturali (seduto/in piedi/accovacciato), non una
    stima di statura (richiederebbe calibrazione della camera).

    Se `normalize=True` (default), il risultato è espresso in "lunghezze di
    busto" (distanza spalle-bacino media), rendendolo confrontabile anche a
    distanze diverse dalla camera; altrimenti è in pixel.
    """
    centers_y = np.array([hip_center(f)[1] for f in kpts_sequence])
    valid = centers_y[~np.isnan(centers_y)]
    if len(valid) == 0:
        return np.nan
    excursion = float(valid.max() - valid.min())
    if not normalize:
        return excursion

    lengths = np.array([torso_length(f) for f in kpts_sequence])
    valid_lengths = lengths[~np.isnan(lengths) & (lengths > 1e-6)]
    if len(valid_lengths) == 0:
        return np.nan
    return excursion / float(np.mean(valid_lengths))


def activity_ratio(energy_series: np.ndarray, threshold: float) -> float:
    """Frazione di frame con energia di movimento sopra `threshold`: proxy
    di quanto tempo la persona passa in movimento vs. relativamente ferma.
    La soglia va calibrata sul contesto (unità = somma dei quadrati delle
    velocità per keypoint, la stessa di `movement_energy`); non esiste un
    valore universalmente corretto.
    """
    valid = energy_series[~np.isnan(energy_series)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid > threshold))


def self_touch_score(wrist_xy: np.ndarray, head_xy: np.ndarray, scale: float) -> float:
    """Punteggio 0-1 di quanto un polso è vicino alla testa, come proxy di
    auto-contatto (mano al volto/capo). Normalizzato sulla scala della
    persona (`torso_length`, o larghezza spalle) così il punteggio è
    confrontabile indipendentemente dalla distanza dalla camera.

    1.0 = polso a contatto con la testa, 0.0 = distanza >= 1 unità di scala.
    Non distingue il motivo del contatto (autoregolazione, prurito,
    comportamento autolesivo, ecc.) — è solo un indicatore di frequenza.
    """
    if scale < 1e-6 or np.isnan(scale):
        return np.nan
    d = np.linalg.norm(wrist_xy - head_xy) / scale
    return float(max(0.0, 1.0 - d))


# ---------------------------------------------------------------------------
# Orchestrazione: costruzione della tabella feature per una persona
# ---------------------------------------------------------------------------

@dataclass
class PersonFeatureTable:
    """Tabella tidy (una riga per frame) di feature per una persona tracciata."""
    track_id: int
    frame_index: np.ndarray
    angles: pd.DataFrame
    speed: pd.DataFrame
    energy: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        df = self.angles.copy()
        df["frame"] = self.frame_index
        df["track_id"] = self.track_id
        df["movement_energy"] = self.energy
        return df


def build_person_features(kpts_sequence: np.ndarray, track_id: int, fps: float) -> PersonFeatureTable:
    """Costruisce la tabella di feature per una persona a partire dalla
    sequenza dei suoi keypoint (n_frame, 17, 2).
    """
    angles = pd.DataFrame([compute_joint_angles(f) for f in kpts_sequence])
    speed = compute_joint_speed(kpts_sequence, fps)
    energy = movement_energy(kpts_sequence, fps)
    frame_index = np.arange(kpts_sequence.shape[0])
    return PersonFeatureTable(track_id, frame_index, angles, speed, energy)
