"""
QRS detection based entirely on NeuroKit2.

This module keeps the old public API (`detect_qrs`, `compute_heart_rate`) so the
rest of the app can stay simple, but the local custom detector has been removed.
"""

from typing import Optional

import numpy as np


def detect_qrs(signal: np.ndarray, fs: int = 200) -> np.ndarray:
    """
    Detect QRS complexes using NeuroKit2.

    Returns:
        Nx2 int32 array:
        - col 0: sample index of the detected R peak
        - col 1: beat type (always 1 for detected normal-readable peaks)
    """
    if signal is None or len(signal) < max(4, fs * 2):
        return np.empty((0, 2), dtype=np.int32)

    try:
        import neurokit2 as nk

        cleaned = nk.ecg_clean(np.asarray(signal, dtype=float), sampling_rate=fs)
        _, peaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        r_peaks = np.asarray(peaks_info.get("ECG_R_Peaks", []), dtype=np.int32)
        if r_peaks.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        markers = np.column_stack(
            [r_peaks, np.ones(r_peaks.size, dtype=np.int32)]
        )
        return markers.astype(np.int32, copy=False)
    except Exception:
        return np.empty((0, 2), dtype=np.int32)


def compute_heart_rate(markers: np.ndarray, fs: int = 200) -> dict:
    """
    Compute heart rate statistics from NeuroKit-based markers.
    """
    if markers is None or len(markers) < 2:
        return {"hr_mean": 0, "hr_std": 0, "hr_min": 0, "hr_max": 0, "rr_intervals": []}

    arr = np.asarray(markers)
    if arr.ndim != 2 or arr.shape[1] < 1:
        return {"hr_mean": 0, "hr_std": 0, "hr_min": 0, "hr_max": 0, "rr_intervals": []}

    positions = np.asarray(arr[:, 0], dtype=np.int32)
    if arr.shape[1] >= 2:
        beat_types = np.asarray(arr[:, 1], dtype=np.int32)
        positions = positions[beat_types == 1]

    if len(positions) < 2:
        return {"hr_mean": 0, "hr_std": 0, "hr_min": 0, "hr_max": 0, "rr_intervals": []}

    rr_intervals = np.diff(positions)
    rr_intervals = rr_intervals[rr_intervals > 0]
    if len(rr_intervals) == 0:
        return {"hr_mean": 0, "hr_std": 0, "hr_min": 0, "hr_max": 0, "rr_intervals": []}

    hr_values = (fs * 60.0) / rr_intervals
    valid = (hr_values > 20) & (hr_values < 300)
    hr_values = hr_values[valid]
    rr_intervals = rr_intervals[valid]

    if len(hr_values) == 0:
        return {"hr_mean": 0, "hr_std": 0, "hr_min": 0, "hr_max": 0, "rr_intervals": []}

    return {
        "hr_mean": float(np.mean(hr_values)),
        "hr_std": float(np.std(hr_values)),
        "hr_min": float(np.min(hr_values)),
        "hr_max": float(np.max(hr_values)),
        "rr_intervals": rr_intervals.tolist(),
    }
