"""
ECG Quality Analysis for Sentinel/Holter 2-lead recordings.
Adapted from Ecg-Interpretation-Python-Service quality.py
with HOLTER_THRESHOLDS for 200 Hz 2-lead signals.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

logger = logging.getLogger(__name__)

# Holter-specific thresholds for 200 Hz 2-lead recordings
HOLTER_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Muscle_Artifact": (0.045, 0.10),
    "Bad_Electrode_Contact": (10, 800),
    "Powerline_Interference": (0.01, 0.05),
    "Baseline_Drift": (0.03, 0.90),
    "Low_SNR": (22, 12),
}

FLAGS_WEIGHTS: Dict[str, float] = {
    "Muscle_Artifact": 0.2,
    "Bad_Electrode_Contact": 0.25,
    "Powerline_Interference": 0.15,
    "Baseline_Drift": 0.2,
    "Low_SNR": 0.2,
    "NK_Peak_Consistency": 0.10,
}

FLAG_MESSAGES: Dict[str, str] = {
    "Muscle_Artifact": "Excess muscle noise",
    "Bad_Electrode_Contact": "Poor electrode contact",
    "Powerline_Interference": "Power-line interference detected",
    "Baseline_Drift": "Baseline drift present",
    "Low_SNR": "Low signal-to-noise ratio",
    "NK_Peak_Consistency": "NK peak sequence inconsistent",
}

_FALLBACK_CONFIG: Dict[str, Any] = {
    "_preset_name": "hardcoded_default",
    "thresholds": dict(HOLTER_THRESHOLDS),
    "flags_weights": dict(FLAGS_WEIGHTS),
    "neurokit": {"enabled": False, "method": "averageQRS", "weight": 0.0},
    "grade_thresholds": {"good": 0.85, "questionable": 0.65},
    "window": {"length_sec": 5.0, "step_sec": 1.0},
    "lead_aggregation": {
        "readable_threshold": 0.65,
        "mean_weight": 0.55,
        "p25_weight": 0.25,
        "coverage_weight": 0.20,
    },
}


def load_quality_config(preset: str = None) -> Dict[str, Any]:
    """Load quality config for a single preset. Falls back to hardcoded defaults on any error."""
    presets_dir = _get_presets_dir()
    try:
        if preset is None:
            meta = _load_meta()
            preset = meta.get("default", "holter_200hz")
        preset_path = presets_dir / f"{preset}.json"
        preset_data = json.loads(preset_path.read_text(encoding="utf-8"))
        # Deep merge: fallback provides defaults, preset overrides per-key at each level
        merged = copy.deepcopy(_FALLBACK_CONFIG)
        for key, value in preset_data.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
        merged["_preset_name"] = preset
        return merged
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load quality config (%s), using fallback defaults", exc)
        return copy.deepcopy(_FALLBACK_CONFIG)


def normalize_quality_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return config with fallback defaults filled in for all optional sections."""
    merged = copy.deepcopy(_FALLBACK_CONFIG)
    if not config:
        return merged

    for key, value in config.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value
    if "_preset_name" in config:
        merged["_preset_name"] = config["_preset_name"]
    return merged


def _get_presets_dir() -> Path:
    """Return the path to the presets directory."""
    return Path(__file__).resolve().parent.parent / "config" / "presets"


def _load_meta() -> Dict[str, Any]:
    """Load _meta.json from presets directory."""
    meta_path = _get_presets_dir() / "_meta.json"
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"default": "holter_200hz"}


def load_all_presets() -> Dict[str, Any]:
    """Load all presets from individual JSON files in the presets directory.
    Returns the same structure as before: {"default": "...", "presets": {...}}.
    On error, returns a minimal structure with _FALLBACK_CONFIG as the single preset."""
    presets_dir = _get_presets_dir()
    try:
        meta = _load_meta()
        presets = {}
        for preset_file in sorted(presets_dir.glob("*.json")):
            if preset_file.name.startswith("_"):
                continue
            name = preset_file.stem
            presets[name] = json.loads(preset_file.read_text(encoding="utf-8"))
        if not presets:
            raise ValueError("No preset files found in config/presets/")
        return {"default": meta.get("default", "holter_200hz"), "presets": presets}
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Could not load presets (%s), using fallback", exc)
        return {
            "default": "hardcoded_default",
            "presets": {"hardcoded_default": copy.deepcopy(_FALLBACK_CONFIG)},
        }


def save_preset(name: str, preset_data: Dict[str, Any]) -> None:
    """Write a single preset to its own JSON file. Uses write-to-temp + rename for safety."""
    presets_dir = _get_presets_dir()
    presets_dir.mkdir(parents=True, exist_ok=True)
    preset_path = presets_dir / f"{name}.json"
    tmp_path = preset_path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(preset_data, indent=2), encoding="utf-8")
        tmp_path.replace(preset_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_presets(data: Dict[str, Any]) -> None:
    """Write all presets from the full structure to individual files.
    Also updates _meta.json with the default preset name."""
    presets_dir = _get_presets_dir()
    presets_dir.mkdir(parents=True, exist_ok=True)
    # Save meta
    meta = {k: v for k, v in data.items() if k != "presets"}
    meta_path = presets_dir / "_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Save each preset as individual file
    for name, preset_data in data.get("presets", {}).items():
        save_preset(name, preset_data)


def analyze_lead_quality(
    signal: np.ndarray,
    sampling_rate: int = 200,
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Compute signal-quality metrics for a single filtered ECG segment.
    NeuroKit2 quality is computed separately on the full signal in analyze_holter_quality().
    """
    t_grades = thresholds if thresholds is not None else HOLTER_THRESHOLDS

    sig = signal - np.mean(signal)
    freqs, psd = welch(sig, fs=sampling_rate)
    total_power = np.sum(psd) + 1e-12

    flags: Dict[str, float] = {}

    nperseg = min(int(2.5 * sampling_rate), len(sig))
    if nperseg < 4:
        nperseg = len(sig)
    freqs2, psd2 = welch(sig, fs=sampling_rate, nperseg=nperseg)
    total_power2 = np.sum(psd2) + 1e-12

    # Powerline interference (48-52 and 58-62 Hz)
    mains_bins = ((freqs2 >= 48) & (freqs2 <= 52)) | ((freqs2 >= 58) & (freqs2 <= 62))
    mains_power = np.sum(psd2[mains_bins])
    pi = mains_power / total_power2
    flags["Powerline_Interference"] = float(
        np.clip(
            (pi - t_grades["Powerline_Interference"][0])
            / (
                t_grades["Powerline_Interference"][1]
                - t_grades["Powerline_Interference"][0]
            ),
            0,
            1,
        )
    )

    # Muscle artifact (35-100 Hz, capped by Nyquist)
    hf_bins = (freqs2 >= 35) & (freqs2 <= min(100, sampling_rate / 2))
    hf_power = np.sum(psd2[hf_bins])
    ma_ratio = hf_power / total_power2
    flags["Muscle_Artifact"] = float(
        np.clip(
            (ma_ratio - t_grades["Muscle_Artifact"][0])
            / (t_grades["Muscle_Artifact"][1] - t_grades["Muscle_Artifact"][0]),
            0,
            1,
        )
    )

    # Baseline drift (<0.5 Hz)
    lf = np.sum(psd[freqs < 0.5])
    bd = lf / total_power
    flags["Baseline_Drift"] = float(
        np.clip(
            (bd - t_grades["Baseline_Drift"][0])
            / (t_grades["Baseline_Drift"][1] - t_grades["Baseline_Drift"][0]),
            0,
            1,
        )
    )

    # QRS amplitude
    amp = float(np.ptp(sig))

    # Bad electrode contact
    if amp < t_grades["Bad_Electrode_Contact"][0]:
        flags["Bad_Electrode_Contact"] = 1.0
    elif amp > t_grades["Bad_Electrode_Contact"][1]:
        flags["Bad_Electrode_Contact"] = 1.0
    else:
        flags["Bad_Electrode_Contact"] = 0.0

    # Low SNR should reflect whether visible QRS energy clearly stands out from
    # high-frequency noise on the already filtered display signal.
    try:
        signal_hi = min(25, sampling_rate / 2 - 1)
        noise_hi = min(80, sampling_rate / 2 - 1)
        noise_lo = min(35, max(10, noise_hi - 5))
        qrs_sos = butter(2, [5, signal_hi], btype="bandpass", fs=sampling_rate, output="sos")
        noise_sos = butter(2, [noise_lo, noise_hi], btype="bandpass", fs=sampling_rate, output="sos")
        qrs_band = sosfiltfilt(qrs_sos, sig)
        noise_band = sosfiltfilt(noise_sos, sig)

        qrs_amp = float(np.percentile(qrs_band, 95) - np.percentile(qrs_band, 5))
        noise_rms = float(np.sqrt(np.mean(noise_band**2)) + 1e-12)
        snr = 20 * np.log10(qrs_amp / noise_rms) if noise_rms > 0 else 60.0

        if qrs_amp < 5:
            flags["Low_SNR"] = 1.0
        else:
            flags["Low_SNR"] = float(
                np.clip(
                    (snr - t_grades["Low_SNR"][0])
                    / (t_grades["Low_SNR"][1] - t_grades["Low_SNR"][0]),
                    0,
                    1,
                )
            )
    except Exception:
        qrs_amp = 0.0
        noise_rms = float(np.sqrt(np.mean(sig**2)) + 1e-12)
        snr = 0.0
        flags["Low_SNR"] = 1.0

    return {
        "flags": flags,
        "values": {
            "m_a": ma_ratio,
            "b_e_c": amp,
            "p_i": pi,
            "b_d": bd,
            "snr": snr,
            "qrs_amp": amp,
            "qrs_band_amp": qrs_amp,
            "hf_noise_rms": noise_rms,
        },
    }


def compute_quality_score(
    flags: Dict[str, float],
    nk_quality: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Derive quality score (0.0-1.0) where each flag deducts weighted penalty.
    Optionally blends with NeuroKit quality index when available."""
    weights = config["flags_weights"] if config and "flags_weights" in config else FLAGS_WEIGHTS
    psd_score = 1.0
    for flag, value in flags.items():
        if value > 0.0:
            psd_score -= weights.get(flag, 0.2) * value
    psd_score = max(0.0, min(1.0, psd_score))

    if nk_quality is not None and config and config.get("neurokit", {}).get("enabled", False):
        nk_weight = config["neurokit"]["weight"]
        score = psd_score * (1.0 - nk_weight) + nk_quality * nk_weight
    else:
        score = psd_score

    return max(0.0, min(1.0, score))


def _iter_window_bounds(n_samples: int, wlen: int, step: int) -> list[Tuple[int, int]]:
    """Build sliding-window [start, end) bounds and include the tail once."""
    if n_samples <= 0 or wlen <= 0 or step <= 0 or n_samples < wlen:
        return []

    bounds: list[Tuple[int, int]] = []
    start = 0
    while start + wlen <= n_samples:
        bounds.append((start, start + wlen))
        start += step

    tail_start = max(0, n_samples - wlen)
    if not bounds or bounds[-1][0] != tail_start:
        bounds.append((tail_start, tail_start + wlen))
    return bounds


def _aggregate_lead_windows(
    window_results: list[Dict[str, Any]],
    lead_key: str,
    agg_cfg: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Aggregate per-window lead scores into one final 0-1 readability score."""
    scores = np.asarray([w[lead_key] for w in window_results], dtype=float)
    if scores.size == 0:
        return 0.0, {"mean": 0.0, "median": 0.0, "p25": 0.0, "coverage": 0.0}

    readable_threshold = float(agg_cfg.get("readable_threshold", 0.65))
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    p25_score = float(np.percentile(scores, 25))
    coverage = float(np.mean(scores >= readable_threshold))

    raw_weights = {
        "mean": max(0.0, float(agg_cfg.get("mean_weight", 0.55))),
        "p25": max(0.0, float(agg_cfg.get("p25_weight", 0.25))),
        "coverage": max(0.0, float(agg_cfg.get("coverage_weight", 0.20))),
    }
    weight_sum = sum(raw_weights.values()) or 1.0
    final_score = (
        mean_score * raw_weights["mean"]
        + p25_score * raw_weights["p25"]
        + coverage * raw_weights["coverage"]
    ) / weight_sum

    return max(0.0, min(1.0, float(final_score))), {
        "mean": mean_score,
        "median": median_score,
        "p25": p25_score,
        "coverage": coverage,
    }


def _compute_nk_peak_consistency(
    rpeak_positions: np.ndarray,
    sampling_rate: int,
    duration_sec: float,
) -> Tuple[float, Dict[str, float]]:
    """Estimate whether NK peak sequence looks physiologically plausible.

    This is a quality proxy for detector stability, not a rhythm normality check.
    We penalize obvious false-positive patterns (very short RR and duplicate-like RR),
    but we do not punish long or irregular RR because those may reflect arrhythmia.
    """
    peak_count = int(len(rpeak_positions))
    if peak_count == 0:
        return 1.0, {
            "peak_count": 0.0,
            "rr_count": 0.0,
            "rr_median_sec": 0.0,
            "rr_short_ratio": 1.0,
            "rr_duplicate_ratio": 1.0,
            "peak_density_bpm": 0.0,
            "peak_span_ratio": 0.0,
            "long_gap_ratio": 1.0,
            "max_gap_sec": max(0.0, duration_sec),
        }

    if peak_count == 1:
        severity = 0.8 if duration_sec >= 20.0 else 0.5
        return severity, {
            "peak_count": 1.0,
            "rr_count": 0.0,
            "rr_median_sec": 0.0,
            "rr_short_ratio": 0.0,
            "rr_duplicate_ratio": severity,
            "peak_density_bpm": 60.0 / max(duration_sec, 1e-6),
            "peak_span_ratio": 0.0,
            "long_gap_ratio": 1.0,
            "max_gap_sec": max(0.0, duration_sec),
        }

    rr_sec = np.diff(np.asarray(rpeak_positions, dtype=np.int32)) / sampling_rate
    rr_sec = rr_sec[rr_sec > 0]
    if rr_sec.size == 0:
        return 1.0, {
            "peak_count": float(peak_count),
            "rr_count": 0.0,
            "rr_median_sec": 0.0,
            "rr_short_ratio": 1.0,
            "rr_duplicate_ratio": 1.0,
            "peak_density_bpm": peak_count * 60.0 / max(duration_sec, 1e-6),
            "peak_span_ratio": 0.0,
            "long_gap_ratio": 1.0,
            "max_gap_sec": max(0.0, duration_sec),
        }

    rr_median_sec = float(np.median(rr_sec))
    short_rr_ratio = float(np.mean(rr_sec < 0.24))
    duplicate_threshold = max(0.24, 0.55 * rr_median_sec)
    duplicate_ratio = float(np.mean(rr_sec < duplicate_threshold))
    peak_density_bpm = peak_count * 60.0 / max(duration_sec, 1e-6)
    peak_span_ratio = float(
        np.clip(
            (float(rpeak_positions[-1]) - float(rpeak_positions[0])) / max(duration_sec * sampling_rate, 1e-6),
            0.0,
            1.0,
        )
    )
    edge_gaps = np.array(
        [
            float(rpeak_positions[0]) / sampling_rate,
            max(0.0, duration_sec - float(rpeak_positions[-1]) / sampling_rate),
        ],
        dtype=float,
    )
    gaps_sec = np.concatenate([edge_gaps[:1], rr_sec, edge_gaps[1:]])
    long_gap_threshold = max(3.0, 3.0 * rr_median_sec)
    long_gap_excess = np.maximum(gaps_sec - long_gap_threshold, 0.0)
    long_gap_ratio = float(np.sum(long_gap_excess) / max(duration_sec, 1e-6))
    max_gap_sec = float(np.max(gaps_sec))

    density_penalty = float(np.clip((24.0 - peak_density_bpm) / 12.0, 0.0, 1.0))
    span_penalty = float(np.clip((0.75 - peak_span_ratio) / 0.75, 0.0, 1.0))
    long_gap_penalty = float(np.clip(long_gap_ratio / 0.20, 0.0, 1.0))
    max_gap_penalty = float(np.clip((max_gap_sec - 4.0) / 4.0, 0.0, 1.0))
    severity = float(
        np.clip(
            0.20 * short_rr_ratio
            + 0.20 * duplicate_ratio
            + 0.25 * density_penalty
            + 0.20 * span_penalty
            + 0.10 * long_gap_penalty
            + 0.05 * max_gap_penalty,
            0.0,
            1.0,
        )
    )
    return severity, {
        "peak_count": float(peak_count),
        "rr_count": float(rr_sec.size),
        "rr_median_sec": rr_median_sec,
        "rr_short_ratio": short_rr_ratio,
        "rr_duplicate_ratio": duplicate_ratio,
        "peak_density_bpm": peak_density_bpm,
        "peak_span_ratio": peak_span_ratio,
        "long_gap_ratio": long_gap_ratio,
        "max_gap_sec": max_gap_sec,
    }


def analyze_holter_quality(
    lead1: np.ndarray,
    lead2: np.ndarray,
    sampling_rate: int = 200,
    window_sec: float = 5,
    step_sec: Optional[float] = None,
    preset: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    markers1: Optional[np.ndarray] = None,
    markers2: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Analyze quality of 2-lead Holter recording.
    Segments into sliding windows and aggregates all windows for per-lead metrics.
    Best-window scores are kept separately for diagnostics.
    NeuroKit2 quality runs on the full signal and blends with PSD at the end.
    Uses best-lead selection (max) instead of weighted average.

    markers1/markers2: optional QRS marker arrays from detect_qrs() (Nx2, col0=pos, col1=type).
    They are kept for compatibility/debugging, but final quality no longer compares
    C# markers against NeuroKit peaks.

    When `config` is provided, uses it directly instead of loading from file.
    When `config` is None, falls back to `load_quality_config(preset)`.
    """
    if config is None:
        config = load_quality_config(preset)
    config = normalize_quality_config(config)

    window_cfg = config.get("window", {})
    effective_window_sec = float(window_cfg.get("length_sec", window_sec))
    effective_step_sec = float(
        step_sec if step_sec is not None else window_cfg.get("step_sec", 1.0)
    )

    wlen = int(effective_window_sec * sampling_rate)
    step = int(effective_step_sec * sampling_rate)
    n = min(len(lead1), len(lead2))
    window_bounds = _iter_window_bounds(n, wlen, step)
    nwin = len(window_bounds)

    if nwin == 0:
        _empty_values = {
            "m_a": 0.0,
            "b_e_c": 0.0,
            "p_i": 0.0,
            "b_d": 0.0,
            "snr": 0.0,
            "qrs_amp": 0.0,
            "nk_quality": None,
            "nk_r_peaks_count": 0,
            "nk_rr_median_sec": 0.0,
            "nk_rr_short_ratio": 0.0,
            "nk_rr_duplicate_ratio": 0.0,
            "nk_peak_density_bpm": 0.0,
            "nk_peak_span_ratio": 0.0,
            "nk_long_gap_ratio": 0.0,
            "nk_max_gap_sec": 0.0,
        }
        return {
            "lead1_quality": 0.0,
            "lead2_quality": 0.0,
            "lead1_psd_quality": 0.0,
            "lead2_psd_quality": 0.0,
            "lead1_psd_best_quality": 0.0,
            "lead2_psd_best_quality": 0.0,
            "overall_quality": 0.0,
            "grade": "Not usable",
            "lead1_flags": {},
            "lead2_flags": {},
            "lead1_values": dict(_empty_values),
            "lead2_values": dict(_empty_values),
            "window_scores": [],
            "best_window_start": 0,
            "best_window_end": 0,
            "quality_best_lead": 1,
            "lead1_nk_quality": None,
            "lead2_nk_quality": None,
            "lead1_window_stats": {
                "mean": 0.0,
                "median": 0.0,
                "p25": 0.0,
                "coverage": 0.0,
            },
            "lead2_window_stats": {
                "mean": 0.0,
                "median": 0.0,
                "p25": 0.0,
                "coverage": 0.0,
            },
            "window_length_sec": effective_window_sec,
            "window_step_sec": effective_step_sec,
            "preset": config.get("_preset_name", "default"),
        }

    window_results = []
    for i, (start, end) in enumerate(window_bounds):
        seg1 = lead1[start:end]
        seg2 = lead2[start:end]

        m1 = analyze_lead_quality(seg1, sampling_rate, thresholds=config["thresholds"])
        m2 = analyze_lead_quality(seg2, sampling_rate, thresholds=config["thresholds"])

        q1 = compute_quality_score(m1["flags"], config=config)
        q2 = compute_quality_score(m2["flags"], config=config)
        overall = max(q1, q2)

        window_results.append(
            {
                "window": i,
                "start_sample": start,
                "end_sample": end,
                "start_sec": start / sampling_rate,
                "end_sec": end / sampling_rate,
                "lead1_score": q1,
                "lead2_score": q2,
                "overall": overall,
                "lead1_flags": m1["flags"],
                "lead2_flags": m2["flags"],
                "lead1_values": m1["values"],
                "lead2_values": m2["values"],
            }
        )

    # All-window PSD means are kept for diagnostics and compatibility.
    q1_psd_all = float(np.mean([w["lead1_score"] for w in window_results]))
    q2_psd_all = float(np.mean([w["lead2_score"] for w in window_results]))

    # Find best consecutive window(s) (~5 sec total) for diagnostics
    best_start = 0
    best_quality = -1.0
    windows_needed = max(1, int(round(5.0 / max(effective_step_sec, 1e-6))))
    for start in range(max(1, nwin - windows_needed + 1)):
        seq = window_results[start : start + windows_needed]
        avg_q = np.mean([w["overall"] for w in seq])
        if avg_q > best_quality:
            best_quality = avg_q
            best_start = start

    best_windows = window_results[best_start : best_start + windows_needed]
    q1_psd_best = float(np.mean([w["lead1_score"] for w in best_windows]))
    q2_psd_best = float(np.mean([w["lead2_score"] for w in best_windows]))

    # Aggregate flags from ALL windows (not just best)
    agg_flags1 = {}
    agg_flags2 = {}
    for flag in FLAG_MESSAGES:
        if flag == "NK_Peak_Consistency":
            continue  # computed separately below
        vals1 = [w["lead1_flags"].get(flag, 0) for w in window_results]
        vals2 = [w["lead2_flags"].get(flag, 0) for w in window_results]
        agg_flags1[flag] = float(np.mean(vals1))
        agg_flags2[flag] = float(np.mean(vals2))

    # Aggregate raw measurement values from ALL windows
    _value_keys = [
        "m_a",
        "b_e_c",
        "p_i",
        "b_d",
        "snr",
        "qrs_amp",
        "qrs_band_amp",
        "hf_noise_rms",
    ]
    lead1_values_agg = {}
    lead2_values_agg = {}
    for key in _value_keys:
        lead1_values_agg[key] = float(np.mean([w["lead1_values"][key] for w in window_results]))
        lead2_values_agg[key] = float(np.mean([w["lead2_values"][key] for w in window_results]))

    # NeuroKit2 quality on full signals (not windowed)
    nk_cfg = config.get("neurokit", {})
    lead1_nk_quality = None
    lead2_nk_quality = None
    lead1_nk_r_peaks = 0
    lead2_nk_r_peaks = 0
    lead1_nk_positions = np.empty(0, dtype=np.int32)
    lead2_nk_positions = np.empty(0, dtype=np.int32)
    if nk_cfg.get("enabled", False):
        nk_min_samples = sampling_rate * 4  # ecg_segment() minimum
        for lead_sig, lead_num in [(lead1[:n], 1), (lead2[:n], 2)]:
            if len(lead_sig) < nk_min_samples:
                logger.debug(
                    "Lead %d too short for NeuroKit2 (%d < %d), skipping",
                    lead_num, len(lead_sig), nk_min_samples,
                )
                continue
            try:
                import neurokit2 as nk

                cleaned = nk.ecg_clean(lead_sig, sampling_rate=sampling_rate)
                _, peaks_info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
                r_peaks = list(peaks_info.get("ECG_R_Peaks", []))
                quality_arr = nk.ecg_quality(
                    cleaned,
                    rpeaks=peaks_info.get("ECG_R_Peaks"),
                    sampling_rate=sampling_rate,
                    method=nk_cfg.get("method", "averageQRS"),
                )
                quality_arr = np.asarray(quality_arr, dtype=float)
                nk_q = float(np.clip(np.nanmean(quality_arr), 0.0, 1.0))
                if not np.isfinite(nk_q):
                    raise ValueError("NeuroKit quality is NaN/Inf")
                if lead_num == 1:
                    lead1_nk_quality = nk_q
                    lead1_nk_r_peaks = len(r_peaks)
                    lead1_nk_positions = np.asarray(r_peaks, dtype=np.int32)
                else:
                    lead2_nk_quality = nk_q
                    lead2_nk_r_peaks = len(r_peaks)
                    lead2_nk_positions = np.asarray(r_peaks, dtype=np.int32)
            except Exception as exc:
                logger.warning("NeuroKit2 quality computation failed for lead %d: %s", lead_num, exc)

    # NK peak consistency — use only NeuroKit peaks and RR plausibility.
    nk_peak_weight = config.get("flags_weights", {}).get(
        "NK_Peak_Consistency",
        FLAGS_WEIGHTS.get("NK_Peak_Consistency", 0.10),
    )
    lead1_values_agg["nk_rr_median_sec"] = 0.0
    lead2_values_agg["nk_rr_median_sec"] = 0.0
    lead1_values_agg["nk_rr_short_ratio"] = 0.0
    lead2_values_agg["nk_rr_short_ratio"] = 0.0
    lead1_values_agg["nk_rr_duplicate_ratio"] = 0.0
    lead2_values_agg["nk_rr_duplicate_ratio"] = 0.0
    lead1_values_agg["nk_peak_density_bpm"] = 0.0
    lead2_values_agg["nk_peak_density_bpm"] = 0.0
    lead1_values_agg["nk_peak_span_ratio"] = 0.0
    lead2_values_agg["nk_peak_span_ratio"] = 0.0
    lead1_values_agg["nk_long_gap_ratio"] = 0.0
    lead2_values_agg["nk_long_gap_ratio"] = 0.0
    lead1_values_agg["nk_max_gap_sec"] = 0.0
    lead2_values_agg["nk_max_gap_sec"] = 0.0
    duration_sec = n / sampling_rate if sampling_rate > 0 else 0.0
    for lead_nk_peaks, lead_nk_positions, lead_num in [
        (lead1_nk_r_peaks, lead1_nk_positions, 1),
        (lead2_nk_r_peaks, lead2_nk_positions, 2),
    ]:
        severity, peak_info = _compute_nk_peak_consistency(
            lead_nk_positions,
            sampling_rate,
            duration_sec,
        )
        if lead_num == 1:
            lead1_values_agg["nk_rr_median_sec"] = float(peak_info["rr_median_sec"])
            lead1_values_agg["nk_rr_short_ratio"] = float(peak_info["rr_short_ratio"])
            lead1_values_agg["nk_rr_duplicate_ratio"] = float(peak_info["rr_duplicate_ratio"])
            lead1_values_agg["nk_peak_density_bpm"] = float(peak_info["peak_density_bpm"])
            lead1_values_agg["nk_peak_span_ratio"] = float(peak_info["peak_span_ratio"])
            lead1_values_agg["nk_long_gap_ratio"] = float(peak_info["long_gap_ratio"])
            lead1_values_agg["nk_max_gap_sec"] = float(peak_info["max_gap_sec"])
        else:
            lead2_values_agg["nk_rr_median_sec"] = float(peak_info["rr_median_sec"])
            lead2_values_agg["nk_rr_short_ratio"] = float(peak_info["rr_short_ratio"])
            lead2_values_agg["nk_rr_duplicate_ratio"] = float(peak_info["rr_duplicate_ratio"])
            lead2_values_agg["nk_peak_density_bpm"] = float(peak_info["peak_density_bpm"])
            lead2_values_agg["nk_peak_span_ratio"] = float(peak_info["peak_span_ratio"])
            lead2_values_agg["nk_long_gap_ratio"] = float(peak_info["long_gap_ratio"])
            lead2_values_agg["nk_max_gap_sec"] = float(peak_info["max_gap_sec"])

        if nk_cfg.get("enabled", False):
            agg = agg_flags1 if lead_num == 1 else agg_flags2
            agg["NK_Peak_Consistency"] = severity

            # Apply as post-hoc penalty to the diagnostic PSD mean.
            penalty = nk_peak_weight * severity
            if lead_num == 1:
                q1_psd_all = max(0.0, q1_psd_all - penalty)
            else:
                q2_psd_all = max(0.0, q2_psd_all - penalty)

    # Aggregate per-window scores into a final readability score per lead.
    lead_agg_cfg = config.get("lead_aggregation", {})
    q1_score, q1_window_stats = _aggregate_lead_windows(
        window_results, "lead1_score", lead_agg_cfg
    )
    q2_score, q2_window_stats = _aggregate_lead_windows(
        window_results, "lead2_score", lead_agg_cfg
    )
    if "NK_Peak_Consistency" in agg_flags1:
        q1_score = max(0.0, q1_score - nk_peak_weight * agg_flags1["NK_Peak_Consistency"])
    if "NK_Peak_Consistency" in agg_flags2:
        q2_score = max(0.0, q2_score - nk_peak_weight * agg_flags2["NK_Peak_Consistency"])

    # Include NK fields in aggregated values
    lead1_values_agg["nk_quality"] = lead1_nk_quality
    lead2_values_agg["nk_quality"] = lead2_nk_quality
    lead1_values_agg["nk_r_peaks_count"] = lead1_nk_r_peaks
    lead2_values_agg["nk_r_peaks_count"] = lead2_nk_r_peaks

    # Blend aggregated lead score with NK (whole-signal) for final scores.
    nk_weight = nk_cfg.get("weight", 0.0) if nk_cfg.get("enabled", False) else 0.0
    q1_avg = q1_score
    q2_avg = q2_score
    if lead1_nk_quality is not None and nk_weight > 0:
        q1_avg = q1_score * (1.0 - nk_weight) + lead1_nk_quality * nk_weight
    if lead2_nk_quality is not None and nk_weight > 0:
        q2_avg = q2_score * (1.0 - nk_weight) + lead2_nk_quality * nk_weight
    q1_avg = max(0.0, min(1.0, q1_avg))
    q2_avg = max(0.0, min(1.0, q2_avg))
    overall = max(q1_avg, q2_avg)

    grade_thresholds = config["grade_thresholds"]
    if overall > grade_thresholds["good"]:
        grade = "Good"
    elif overall > grade_thresholds["questionable"]:
        grade = "Questionable"
    else:
        grade = "Not usable"

    return {
        "lead1_quality": q1_avg,
        "lead2_quality": q2_avg,
        "lead1_psd_quality": q1_psd_all,
        "lead2_psd_quality": q2_psd_all,
        "lead1_psd_best_quality": q1_psd_best,
        "lead2_psd_best_quality": q2_psd_best,
        "overall_quality": overall,
        "grade": grade,
        "lead1_flags": agg_flags1,
        "lead2_flags": agg_flags2,
        "lead1_values": lead1_values_agg,
        "lead2_values": lead2_values_agg,
        "best_window_start": best_start,
        "best_window_end": best_start + windows_needed,
        "lead1_window_stats": q1_window_stats,
        "lead2_window_stats": q2_window_stats,
        "window_length_sec": effective_window_sec,
        "window_step_sec": effective_step_sec,
        "quality_best_lead": 1 if q1_avg >= q2_avg else 2,
        "lead1_nk_quality": lead1_nk_quality,
        "lead2_nk_quality": lead2_nk_quality,
        "preset": config.get("_preset_name", "default"),
        "window_scores": [
            {
                "window": w["window"],
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "lead1": w["lead1_score"],
                "lead2": w["lead2_score"],
                "overall": w["overall"],
            }
            for w in window_results
        ],
    }
