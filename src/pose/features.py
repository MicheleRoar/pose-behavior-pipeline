"""
features.py
============
Extraction of behavioral features from 2D keypoint time series (COCO-17
schema), designed as reusable "building blocks" for analyzing
child-caregiver interaction videos.

Design:
- No dependency on a specific pose estimation model: all functions
  operate on numpy arrays (n_frame, 17, 2/3), so they work both with
  Ultralytics YOLO-pose output and with MediaPipe/OpenPose, once remapped
  to the COCO-17 schema.
- Features are chosen to reflect markers discussed in the literature for
  studying child neurodevelopment via video (see README):
    * joint angles and their variability (posture, motor control)
    * movement velocity/energy
    * left/right symmetry index
    * repetitive movement "score" (stereotypies) via autocorrelation
    * proximity and synchrony between two tracked people (child/caregiver)

Methodological note: these features are an exploratory starting point,
not validated diagnostic markers. Any clinical use requires validation on
annotated data and supervision by qualified personnel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pose.keypoints import KP, LR_PAIRS, JOINT_ANGLE_TRIPLETS
from pose.geometry import angle_at as _angle_at


def compute_joint_angles(frame_kpts: np.ndarray) -> dict[str, float]:
    """Computes the joint angles defined in JOINT_ANGLE_TRIPLETS for a
    single frame.

    Parameters
    ----------
    frame_kpts : array (17, 2) with the (x, y) coordinates of the COCO-17 keypoints.
    """
    angles = {}
    for name, (a_name, b_name, c_name) in JOINT_ANGLE_TRIPLETS.items():
        a, b, c = frame_kpts[KP[a_name]], frame_kpts[KP[b_name]], frame_kpts[KP[c_name]]
        angles[name] = _angle_at(a, b, c)
    return angles


# ---------------------------------------------------------------------------
# Kinematics: velocity and movement energy
# ---------------------------------------------------------------------------

def compute_joint_speed(kpts_sequence: np.ndarray, fps: float) -> pd.DataFrame:
    """Velocity (units/second) of each keypoint along the sequence.

    Parameters
    ----------
    kpts_sequence : array (n_frame, 17, 2)
    fps : video frame rate

    Returns
    -------
    DataFrame (n_frame, 17) with the instantaneous velocity of each
    keypoint (the first frame is NaN, as there's no previous frame).
    """
    diffs = np.diff(kpts_sequence, axis=0) * fps  # units/second
    speed = np.linalg.norm(diffs, axis=2)  # (n_frame-1, 17)
    speed = np.vstack([np.full((1, speed.shape[1]), np.nan), speed])
    return pd.DataFrame(speed, columns=[k for k in KP])


def movement_energy(kpts_sequence: np.ndarray, fps: float) -> np.ndarray:
    """Proxy for overall kinetic energy per frame: sum of squared
    velocities across all keypoints. Useful as a synthetic index of "how
    much" a person is moving at a given instant.
    """
    speed_df = compute_joint_speed(kpts_sequence, fps)
    return (speed_df ** 2).sum(axis=1).to_numpy()


# ---------------------------------------------------------------------------
# Left/right symmetry
# ---------------------------------------------------------------------------

def symmetry_index(kpts_sequence: np.ndarray, fps: float) -> pd.Series:
    """Symmetry index for each left/right pair, defined as

        SI = |v_left - v_right| / (v_left + v_right)

    computed on the average velocity of each limb along the sequence.
    SI = 0 -> perfectly symmetric, SI -> 1 -> strongly asymmetric.
    Conceptual reference: studies on movement asymmetries and midline
    postural control in ASD (see README).
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
# Repetitive movement / stereotypies
# ---------------------------------------------------------------------------

def repetitive_motion_score(signal: np.ndarray, fps: float,
                             min_freq_hz: float = 0.5,
                             max_freq_hz: float = 8.0) -> dict[str, float]:
    """Quantifies how much a 1D signal (e.g. a wrist's velocity) is
    dominated by a periodic oscillation in a frequency band plausible for
    repetitive manual movements (stereotypies).

    Approach: power spectral density (FFT) + ratio between the dominant
    peak's power in the band [min_freq_hz, max_freq_hz] and the signal's
    total power ("peak power ratio"). A high value indicates a strongly
    periodic movement in that band.

    Note: the default band is a reasonable starting point for repetitive
    manual movements; the optimal band should be validated on
    real/annotated data for the specific clinical context.
    """
    signal = np.nan_to_num(signal - np.nanmean(signal))
    n = len(signal)
    if n < 8:
        return {"peak_freq_hz": np.nan, "peak_power_ratio": np.nan}

    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal)) ** 2

    band_mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    total_power = power[1:].sum()  # excludes the DC component
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
# Interaction between two people (e.g. child-caregiver)
# ---------------------------------------------------------------------------

def hip_center(frame_kpts: np.ndarray) -> np.ndarray:
    """Pelvis center, used as a proxy for body position."""
    return (frame_kpts[KP["left_hip"]] + frame_kpts[KP["right_hip"]]) / 2.0


def proximity_series(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Euclidean distance between the hip-centers of two tracked people,
    frame by frame. The two sequences must have the same length and be
    temporally aligned (same frame index).
    """
    centers_a = np.array([hip_center(f) for f in seq_a])
    centers_b = np.array([hip_center(f) for f in seq_b])
    return np.linalg.norm(centers_a - centers_b, axis=1)


def windowed_synchrony(signal_a: np.ndarray, signal_b: np.ndarray,
                        window: int, step: int) -> pd.DataFrame:
    """Pearson correlation between two movement signals (e.g. child's and
    caregiver's kinetic energy) computed over sliding windows, as a
    simple proxy for dyadic motor synchrony.
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
# Posture, activity, and self-touch
# ---------------------------------------------------------------------------

def torso_length(frame_kpts: np.ndarray) -> float:
    """Shoulder-to-hip distance in a frame, used as the person's scale
    unit (invariant to distance from the camera) to normalize other
    features (vertical excursion, self-touch)."""
    shoulder_center = (frame_kpts[KP["left_shoulder"]] + frame_kpts[KP["right_shoulder"]]) / 2.0
    return float(np.linalg.norm(shoulder_center - hip_center(frame_kpts)))


def vertical_excursion(kpts_sequence: np.ndarray, normalize: bool = True) -> float:
    """Vertical excursion (max - min) of the hip-center along the
    sequence: a proxy for postural transitions (sitting/standing/
    crouching), not a height estimate (would require camera calibration).

    If `normalize=True` (default), the result is expressed in "torso
    lengths" (average shoulder-to-hip distance), making it comparable
    even at different distances from the camera; otherwise it's in
    pixels.
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
    """Fraction of frames with movement energy above `threshold`: a proxy
    for how much time the person spends moving vs. relatively still. The
    threshold must be calibrated to the context (units = sum of squared
    velocities per keypoint, same as `movement_energy`); there is no
    universally correct value.
    """
    valid = energy_series[~np.isnan(energy_series)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid > threshold))


def self_touch_score(wrist_xy: np.ndarray, head_xy: np.ndarray, scale: float) -> float:
    """0-1 score of how close a wrist is to the head, as a proxy for
    self-contact (hand to face/head). Normalized by the person's scale
    (`torso_length`, or shoulder width) so the score is comparable
    regardless of distance from the camera.

    1.0 = wrist in contact with the head, 0.0 = distance >= 1 scale unit.
    Doesn't distinguish the reason for the contact (self-regulation,
    itching, self-injurious behavior, etc.) -- it's only a frequency
    indicator.
    """
    if scale < 1e-6 or np.isnan(scale):
        return np.nan
    d = np.linalg.norm(wrist_xy - head_xy) / scale
    return float(max(0.0, 1.0 - d))


# ---------------------------------------------------------------------------
# Orchestration: building the feature table for a person
# ---------------------------------------------------------------------------

@dataclass
class PersonFeatureTable:
    """Tidy table (one row per frame) of features for a tracked person."""
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
    """Builds the feature table for a person from their keypoint sequence
    (n_frame, 17, 2).
    """
    angles = pd.DataFrame([compute_joint_angles(f) for f in kpts_sequence])
    speed = compute_joint_speed(kpts_sequence, fps)
    energy = movement_energy(kpts_sequence, fps)
    frame_index = np.arange(kpts_sequence.shape[0])
    return PersonFeatureTable(track_id, frame_index, angles, speed, energy)
