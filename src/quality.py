"""
ECG Quality Analysis for Sentinel/Holter 2-lead recordings.
Adapted from Ecg-Interpretation-Python-Service quality.py
with HOLTER_THRESHOLDS for 200 Hz 2-lead signals.
"""

"""
Alex fix 
Ошибка  в  формуле Low_SNR, раньше noise считался как разница между сигналом и его же band-pass версией. Для уже отфильтрованного ECG это плохо: часть полезного QRS попадала в “шум”, поэтому даже чистая запись давала заниженный SNR. Из-за этого нормальный лид мог получать Low SNR ~0.289 и хуже.
полезный сигнал: амплитуда в QRS-полосе 5-25 Hz
шум: RMS высокочастотной полосы 35-80 Hz
snr = 20*log10(qrs_band_amp / hf_noise_rms)
если qrs_band_amp слишком маленькая, Low_SNR сразу считается плохим

вместо неперекрывающихся окон теперь используется sliding window с дефолтом 5.0s и шагом 1.0s;
финальный lead1_quality / lead2_quality теперь считается не просто как среднее по окнам, а как смесь из mean + p25 + coverage, чтобы штрафовать лиды, где читаемость нестабильна по времени;
убрано внутреннее повторное notch-подавление внутри quality-метрик, чтобы оценка соответствовала именно тому filtered-сигналу, который видит пользователь;

убрал влияние QRS_Count_Mismatch (Сравнение 2  алгоритмов) на итоговую оценку и заменил его на новый флаг NK_Peak_Consistency. Теперь он считается только по NK R-peaks и RR:
штрафуются только явно подозрительные вещи вроде слишком коротких RR и duplicate-like RR;
длинные и неровные RR сами по себе не считаются плохим качеством, чтобы не наказывать аритмию;
если NK вообще не находит пики, это тоже считается плохим признаком качества.
Добавил штрафы за:
низкую плотность пиков nk_peak_density_bpm
слабое покрытие записи пиками nk_peak_span_ratio
длинные пустые участки без пиков nk_long_gap_ratio
максимальный gap nk_max_gap_sec

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
    "Periodic_Artifact": (0.10, 0.45),
    "Baseline_Drift": (0.03, 0.90),
    "Low_SNR": (22, 12),
}

FLAGS_WEIGHTS: Dict[str, float] = {
    "Noise_Artifact": 0.35,
    "Bad_Electrode_Contact": 0.15,
    "Baseline_Drift": 0.10,
    "Low_SNR": 0.40,
}

FLAG_MESSAGES: Dict[str, str] = {
    "Noise_Artifact": "Residual noise / interference",
    "Bad_Electrode_Contact": "Poor electrode contact",
    "Baseline_Drift": "Baseline drift present",
    "Low_SNR": "Low signal-to-noise ratio",
}

_FALLBACK_CONFIG: Dict[str, Any] = {
    "_preset_name": "hardcoded_default",
    "thresholds": dict(HOLTER_THRESHOLDS),
    "flags_weights": dict(FLAGS_WEIGHTS),
    "neurokit": {"enabled": False, "method": "averageQRS", "weight": 0.0},
    "grade_thresholds": {"good": 0.85, "questionable": 0.65},
    "window": {"length_sec": 5.0, "step_sec": 1.0, "ignore_initial_sec": 3.0},
    "baseline_visibility": {
        "enabled": True,
        "chunk_sec": 0.40,
        "good_ratio": 0.06,
        "bad_ratio": 0.25,
    },
    "readability_relief": {
        "enabled": True,
        "support_threshold": 0.45,
        "support_full": 0.75,
        "morph_threshold": 0.70,
        "morph_full": 0.90,
        "noise_reference": 0.55,
        "low_snr_reference": 0.85,
        "max_bonus": 0.18,
        "legacy_max_bonus": 0.0,
    },
    "lead_aggregation": {
        "readable_threshold": 0.65,
        "mean_weight": 0.60,
        "p25_weight": 0.40,
        "low_gain_bonus": 0.30,
        "low_gain_max_base": 0.45,
        "low_gain_max_std": 0.08,
        "low_gain_min_support": 0.45,
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
    config: Optional[Dict[str, Any]] = None,
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

    # Powerline interference: compare mains energy against useful ECG-band energy,
    # then compress the result to a 0..1 perceptual scale.
    mains_bins = ((freqs2 >= 48) & (freqs2 <= 52)) | ((freqs2 >= 58) & (freqs2 <= 62))
    mains_power = np.sum(psd2[mains_bins])
    ecg_bins = (freqs2 >= 0.5) & (freqs2 <= min(40, sampling_rate / 2 - 1))
    ecg_power = np.sum(psd2[ecg_bins])
    mains_db = float(10 * np.log10((mains_power + 1e-12) / (ecg_power + 1e-12)))
    pi = _map_powerline_db_to_unit(mains_db)
    powerline_flag = pi

    # Periodic narrowband artifact: strong, concentrated peaks in an artifact-heavy
    # high-frequency band. We intentionally start above ~30 Hz to avoid punishing
    # stable generator ECG harmonics around the upper end of the QRS band.
    periodic_bins = (freqs2 >= 30) & (freqs2 <= min(90, sampling_rate / 2 - 1))
    periodic_freqs = freqs2[periodic_bins]
    periodic_psd = psd2[periodic_bins]
    if periodic_psd.size >= 3:
        peak_idx = int(np.argmax(periodic_psd))
        left = max(0, peak_idx - 1)
        right = min(periodic_psd.size, peak_idx + 2)
        local_peak_power = float(np.sum(periodic_psd[left:right]))
        periodic_band_power = float(np.sum(periodic_psd) + 1e-12)
        periodic_ratio = local_peak_power / periodic_band_power
        periodic_freq = float(periodic_freqs[peak_idx])
    else:
        local_peak_power = 0.0
        periodic_ratio = 0.0
        periodic_freq = 0.0

    # Muscle artifact should emphasize broadband HF noise. If most HF energy is
    # concentrated in one narrow line, that belongs more to Periodic_Artifact
    # than to true EMG-like noise.
    hf_bins = (freqs2 >= 35) & (freqs2 <= min(100, sampling_rate / 2))
    hf_power = np.sum(psd2[hf_bins])
    hf_broadband_power = max(0.0, float(hf_power - min(local_peak_power, hf_power)))
    ma_ratio = hf_broadband_power / total_power2
    muscle_flag = float(
        np.clip(
            (ma_ratio - t_grades["Muscle_Artifact"][0])
            / (t_grades["Muscle_Artifact"][1] - t_grades["Muscle_Artifact"][0]),
            0,
            1,
        )
    )

    # Baseline drift (<0.5 Hz) from residual LF power after preprocessing.
    lf = np.sum(psd[freqs < 0.5])
    bd = lf / total_power
    baseline_psd_flag = float(
        np.clip(
            (bd - t_grades["Baseline_Drift"][0])
            / (t_grades["Baseline_Drift"][1] - t_grades["Baseline_Drift"][0]),
            0,
            1,
        )
    )
    # Visible baseline wander: even after HP filtering, the user may still see
    # the baseline moving up/down inside the display window. A robust way to
    # capture this is to compare short-segment medians across the window.
    baseline_vis_cfg = normalize_quality_config(config).get("baseline_visibility", {})
    visible_chunk_sec = float(baseline_vis_cfg.get("chunk_sec", 0.40))
    visible_good_ratio = float(baseline_vis_cfg.get("good_ratio", 0.06))
    visible_bad_ratio = float(baseline_vis_cfg.get("bad_ratio", 0.25))
    visible_chunk = max(1, int(round(visible_chunk_sec * sampling_rate)))
    visible_ratio = 0.0
    if bool(baseline_vis_cfg.get("enabled", True)) and len(sig) >= visible_chunk * 4:
        medians = np.asarray(
            [
                float(np.median(sig[i : i + visible_chunk]))
                for i in range(0, len(sig) - visible_chunk + 1, visible_chunk)
            ],
            dtype=float,
        )
        if medians.size >= 4:
            visible_ratio = float(
                (np.percentile(medians, 95) - np.percentile(medians, 5))
                / (float(np.ptp(sig)) + 1e-12)
            )
    baseline_visible_flag = float(
        np.clip(
            (visible_ratio - visible_good_ratio)
            / (visible_bad_ratio - visible_good_ratio + 1e-12),
            0.0,
            1.0,
        )
    )
    # Final baseline drift uses the stronger of:
    # 1) residual low-frequency power
    # 2) visible baseline movement on the displayed signal
    flags["Baseline_Drift"] = max(baseline_psd_flag, baseline_visible_flag)

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

    periodic_raw = float(
        np.clip(
            (periodic_ratio - t_grades["Periodic_Artifact"][0])
            / (t_grades["Periodic_Artifact"][1] - t_grades["Periodic_Artifact"][0]),
            0,
            1,
        )
    )
    periodic_context = max(muscle_flag, powerline_flag)
    periodic_flag = float(periodic_raw * periodic_context)
    abs_diff = np.abs(np.diff(sig, prepend=sig[0]))
    diff_p95 = float(np.percentile(abs_diff, 95))
    diff_max = float(np.max(abs_diff))
    transient_ratio = diff_max / (diff_p95 + 1e-12)
    transient_flag = float(np.clip((transient_ratio - 3.0) / 2.0, 0.0, 1.0))
    # Keep transient spikes as a diagnostic metric only. In filtered ECG,
    # sharp physiological QRS slopes can look like "spikes" in the derivative
    # domain, so using transient_flag directly in the final noise score tends
    # to punish clean recordings.
    flags["Noise_Artifact"] = float(
        max(muscle_flag, powerline_flag, periodic_flag)
    )

    return {
        "flags": flags,
        "values": {
            "m_a": ma_ratio,
            "muscle_flag_raw": muscle_flag,
            "m_a_hf_power": float(hf_power),
            "m_a_broadband_hf_power": hf_broadband_power,
            "b_e_c": amp,
            "p_i": pi,
            "powerline_flag_raw": powerline_flag,
            "p_i_db": mains_db,
            "noise_artifact_flag": flags["Noise_Artifact"],
            "transient_artifact_ratio": transient_ratio,
            "transient_flag_raw": transient_flag,
            "periodic_artifact_ratio": periodic_ratio,
            "periodic_artifact_freq": periodic_freq,
            "periodic_artifact_context": periodic_context,
            "periodic_flag_raw": periodic_flag,
            "b_d": bd,
            "baseline_psd_flag_raw": baseline_psd_flag,
            "baseline_visible_ratio": visible_ratio,
            "baseline_visible_flag_raw": baseline_visible_flag,
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
    """Derive a simple 0..1 window score from the main readability penalties."""
    cfg = normalize_quality_config(config)
    weights = cfg.get("flags_weights", FLAGS_WEIGHTS)
    positive_weights = {
        flag: max(0.0, float(weight))
        for flag, weight in weights.items()
        if float(weight) > 0.0
    }
    weight_sum = sum(positive_weights.values()) or 1.0
    norm_weights = {flag: value / weight_sum for flag, value in positive_weights.items()}

    clamped_flags = {
        flag: float(np.clip(value, 0.0, 1.0))
        for flag, value in flags.items()
    }

    penalty = 0.0
    for flag, value in clamped_flags.items():
        penalty += norm_weights.get(flag, 0.0) * value
    return max(0.0, min(1.0, 1.0 - penalty))


def _map_powerline_db_to_unit(mains_db: float) -> float:
    """Map mains-vs-ECG ratio in dB to a perceptual 0..1 interference score.

    Anchors:
    - <= -15 dB: 0.00 (mains almost does not matter)
    -  -7.5 dB: 0.33 (noticeable)
    -   0.0 dB: 0.66 (mains comparable to useful ECG energy)
    - >= 10 dB: 1.00 (mains dominates)
    """
    if mains_db <= -15.0:
        return 0.0
    if mains_db <= -7.5:
        return float((mains_db + 15.0) / 7.5 * 0.33)
    if mains_db <= 0.0:
        return float(0.33 + (mains_db + 7.5) / 7.5 * 0.33)
    if mains_db <= 10.0:
        return float(0.66 + mains_db / 10.0 * 0.34)
    return 1.0


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
    std_score = float(np.std(scores))
    coverage = float(np.mean(scores >= readable_threshold))

    raw_weights = {
        "mean": max(0.0, float(agg_cfg.get("mean_weight", 0.60))),
        "p25": max(0.0, float(agg_cfg.get("p25_weight", 0.40))),
    }
    weight_sum = sum(raw_weights.values()) or 1.0
    final_score = (
        mean_score * raw_weights["mean"]
        + p25_score * raw_weights["p25"]
    ) / weight_sum

    return max(0.0, min(1.0, float(final_score))), {
        "mean": mean_score,
        "median": median_score,
        "p25": p25_score,
        "std": std_score,
        "coverage": coverage,
    }


def _compute_readability_support(
    peak_info: Dict[str, float],
    nk_quality: Optional[float],
    snr_db: float,
) -> float:
    """Estimate whether the lead stays readable over the full fragment."""
    peak_consistency = float(peak_info.get("consistency_severity", 1.0))
    peak_span_ratio = float(peak_info.get("peak_span_ratio", 0.0))
    peak_density_bpm = float(peak_info.get("peak_density_bpm", 0.0))
    peak_count = float(peak_info.get("peak_count", 0.0))

    consistency_gate = float(np.clip((0.20 - peak_consistency) / 0.20, 0.0, 1.0))
    span_gate = float(np.clip((peak_span_ratio - 0.85) / 0.10, 0.0, 1.0))
    density_low_gate = float(np.clip((peak_density_bpm - 25.0) / 15.0, 0.0, 1.0))
    density_high_gate = float(np.clip((150.0 - peak_density_bpm) / 40.0, 0.0, 1.0))
    density_gate = min(density_low_gate, density_high_gate)
    nk_quality_gate = (
        1.0
        if nk_quality is None
        else float(np.clip((nk_quality - 0.45) / 0.25, 0.0, 1.0))
    )
    if peak_count < 8:
        return 0.0
    snr_gate = float(np.clip((snr_db - 4.0) / 6.0, 0.0, 1.0))
    return float(
        np.clip(
            0.50 * consistency_gate
            + 0.30 * span_gate
            + 0.20 * density_gate,
            0.0,
            1.0,
        )
        * nk_quality_gate
        * snr_gate
    )


def _compute_peak_unreliability(values: Dict[str, float]) -> float:
    """Simple NK-only penalty for leads whose detected peaks do not cover the record well."""
    consistency = float(values.get("peak_consistency_proxy", 1.0))
    span = float(values.get("nk_peak_span_ratio", 0.0))
    long_gap_ratio = float(values.get("nk_long_gap_ratio", 1.0))
    max_gap_sec = float(values.get("nk_max_gap_sec", 0.0))
    morphology = float(values.get("nk_morph_unreliability", 0.0))

    span_penalty = float(np.clip((0.85 - span) / 0.25, 0.0, 1.0))
    long_gap_penalty = float(np.clip(long_gap_ratio / 0.15, 0.0, 1.0))
    max_gap_penalty = float(np.clip((max_gap_sec - 3.5) / 3.0, 0.0, 1.0))

    return float(
        np.clip(
            max(
                consistency,
                0.80 * span_penalty,
                0.60 * long_gap_penalty,
                0.80 * max_gap_penalty,
                morphology,
            ),
            0.0,
            1.0,
        )
    )


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


def _compute_peak_morphology_unreliability(
    signal: np.ndarray,
    rpeak_positions: np.ndarray,
    sampling_rate: int,
) -> Dict[str, float]:
    """Penalize leads where NK peaks do not form a repeatable ECG-like morphology."""
    sig = np.asarray(signal, dtype=float)
    peaks = np.asarray(rpeak_positions, dtype=np.int32)
    if sig.size == 0 or peaks.size < 4:
        return {
            "peak_to_mad": 0.0,
            "median_peak_abs": 0.0,
            "p25_corr": 0.0,
            "median_corr": 0.0,
            "support": 0.0,
            "unreliability": 1.0,
        }

    valid_peaks = peaks[(peaks >= 0) & (peaks < sig.size)]
    if valid_peaks.size < 4:
        return {
            "peak_to_mad": 0.0,
            "median_peak_abs": 0.0,
            "p25_corr": 0.0,
            "median_corr": 0.0,
            "support": 0.0,
            "unreliability": 1.0,
        }

    signal_median = float(np.median(sig))
    signal_mad = float(np.median(np.abs(sig - signal_median))) + 1e-12
    peak_abs = np.abs(sig[valid_peaks])
    median_peak_abs = float(np.median(peak_abs)) if peak_abs.size else 0.0
    peak_to_mad = median_peak_abs / signal_mad

    pre = max(1, int(round(0.10 * sampling_rate)))
    post = max(2, int(round(0.16 * sampling_rate)))
    beats = []
    for peak in valid_peaks:
        start = int(peak) - pre
        end = int(peak) + post
        if start < 0 or end >= sig.size:
            continue
        beat = sig[start:end].astype(float, copy=False)
        beat = beat - np.mean(beat)
        beat_rms = float(np.sqrt(np.mean(beat**2)))
        if beat_rms < 1e-9:
            continue
        beats.append(beat / beat_rms)

    if len(beats) < 4:
        return {
            "peak_to_mad": peak_to_mad,
            "median_peak_abs": median_peak_abs,
            "p25_corr": 0.0,
            "median_corr": 0.0,
            "support": 0.0,
            "unreliability": 1.0,
        }

    beat_matrix = np.vstack(beats)
    template = np.median(beat_matrix, axis=0)
    template = template - np.mean(template)
    template_rms = float(np.sqrt(np.mean(template**2)))
    if template_rms < 1e-9:
        return {
            "peak_to_mad": peak_to_mad,
            "median_peak_abs": median_peak_abs,
            "p25_corr": 0.0,
            "median_corr": 0.0,
            "support": 0.0,
            "unreliability": 1.0,
        }
    template = template / template_rms

    correlations = []
    for beat in beat_matrix:
        corr = float(np.corrcoef(beat, template)[0, 1])
        if np.isfinite(corr):
            correlations.append(corr)

    if len(correlations) < 4:
        return {
            "peak_to_mad": peak_to_mad,
            "median_peak_abs": median_peak_abs,
            "p25_corr": 0.0,
            "median_corr": 0.0,
            "support": 0.0,
            "unreliability": 1.0,
        }

    corr_arr = np.asarray(correlations, dtype=float)
    p25_corr = float(np.percentile(corr_arr, 25))
    median_corr = float(np.median(corr_arr))
    corr_gate = float(np.clip((p25_corr - 0.60) / 0.25, 0.0, 1.0))
    prominence_gate = float(np.clip((peak_to_mad - 3.0) / 1.5, 0.0, 1.0))
    support = float(min(corr_gate, prominence_gate))
    unreliability = float(np.clip((0.55 - support) / 0.35, 0.0, 1.0))

    return {
        "peak_to_mad": peak_to_mad,
        "median_peak_abs": median_peak_abs,
        "p25_corr": p25_corr,
        "median_corr": median_corr,
        "support": support,
        "unreliability": unreliability,
    }


def _scaled_gate(value: float, start: float, full: float) -> float:
    """Map a metric to 0..1 between two control points."""
    if full <= start:
        return float(value >= full)
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


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

    markers1/markers2 are ignored and kept only for call-site compatibility.
    Peak-based quality logic now relies only on NeuroKit detections.

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
    effective_ignore_initial_sec = float(window_cfg.get("ignore_initial_sec", 3.0))

    wlen = int(effective_window_sec * sampling_rate)
    step = int(effective_step_sec * sampling_rate)
    n = min(len(lead1), len(lead2))
    analysis_start = int(max(0.0, effective_ignore_initial_sec) * sampling_rate)
    if n - analysis_start < wlen:
        analysis_start = 0
    trimmed_n = max(0, n - analysis_start)
    relative_bounds = _iter_window_bounds(trimmed_n, wlen, step)
    window_bounds = [
        (analysis_start + start, analysis_start + end)
        for start, end in relative_bounds
    ]
    nwin = len(window_bounds)

    if nwin == 0:
        _empty_values = {
            "m_a": 0.0,
            "muscle_flag_raw": 0.0,
            "m_a_hf_power": 0.0,
            "m_a_broadband_hf_power": 0.0,
            "b_e_c": 0.0,
            "p_i": 0.0,
            "powerline_flag_raw": 0.0,
            "p_i_db": -30.0,
            "noise_artifact_flag": 0.0,
            "periodic_artifact_ratio": 0.0,
            "periodic_artifact_freq": 0.0,
            "periodic_artifact_context": 0.0,
            "periodic_flag_raw": 0.0,
            "b_d": 0.0,
            "snr": 0.0,
            "qrs_amp": 0.0,
            "qrs_band_amp": 0.0,
            "hf_noise_rms": 0.0,
            "nk_quality": None,
            "nk_r_peaks_count": 0,
            "nk_rr_median_sec": 0.0,
            "nk_rr_short_ratio": 0.0,
            "nk_rr_duplicate_ratio": 0.0,
            "nk_peak_density_bpm": 0.0,
            "nk_peak_span_ratio": 0.0,
            "nk_long_gap_ratio": 0.0,
            "nk_max_gap_sec": 0.0,
            "peak_consistency_proxy": 1.0,
            "peak_consistency_count": 0.0,
            "readability_support": 0.0,
            "readability_bonus": 0.0,
            "window_quality": 0.0,
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
            "analysis_start_sec": analysis_start / sampling_rate if sampling_rate > 0 else 0.0,
            "preset": config.get("_preset_name", "default"),
        }

    window_results = []
    for i, (start, end) in enumerate(window_bounds):
        seg1 = lead1[start:end]
        seg2 = lead2[start:end]

        m1 = analyze_lead_quality(
            seg1,
            sampling_rate,
            thresholds=config["thresholds"],
            config=config,
        )
        m2 = analyze_lead_quality(
            seg2,
            sampling_rate,
            thresholds=config["thresholds"],
            config=config,
        )

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
        vals1 = [w["lead1_flags"].get(flag, 0) for w in window_results]
        vals2 = [w["lead2_flags"].get(flag, 0) for w in window_results]
        agg_flags1[flag] = float(np.mean(vals1))
        agg_flags2[flag] = float(np.mean(vals2))

    # Aggregate raw measurement values from ALL windows
    _value_keys = [
        "m_a",
        "muscle_flag_raw",
        "m_a_hf_power",
        "m_a_broadband_hf_power",
        "b_e_c",
        "p_i",
        "powerline_flag_raw",
        "p_i_db",
        "noise_artifact_flag",
        "transient_artifact_ratio",
        "transient_flag_raw",
        "periodic_artifact_ratio",
        "periodic_artifact_freq",
        "periodic_artifact_context",
        "periodic_flag_raw",
        "b_d",
        "baseline_psd_flag_raw",
        "baseline_visible_ratio",
        "baseline_visible_flag_raw",
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

    # NeuroKit2 detections on full signals (always used for peak logic).
    # Whole-signal quality is computed only when enabled in config.
    nk_cfg = config.get("neurokit", {})
    lead1_nk_quality = None
    lead2_nk_quality = None
    lead1_nk_r_peaks = 0
    lead2_nk_r_peaks = 0
    lead1_nk_positions = np.empty(0, dtype=np.int32)
    lead2_nk_positions = np.empty(0, dtype=np.int32)
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
            nk_q = None
            if nk_cfg.get("enabled", False):
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
            logger.warning("NeuroKit2 detection/quality failed for lead %d: %s", lead_num, exc)

    # NK peak consistency — use only NeuroKit peaks and RR plausibility.
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
    lead1_values_agg["peak_consistency_proxy"] = 1.0
    lead2_values_agg["peak_consistency_proxy"] = 1.0
    lead1_values_agg["peak_consistency_count"] = 0.0
    lead2_values_agg["peak_consistency_count"] = 0.0
    lead1_values_agg["nk_peak_to_mad"] = 0.0
    lead2_values_agg["nk_peak_to_mad"] = 0.0
    lead1_values_agg["nk_morph_p25_corr"] = 0.0
    lead2_values_agg["nk_morph_p25_corr"] = 0.0
    lead1_values_agg["nk_morph_median_corr"] = 0.0
    lead2_values_agg["nk_morph_median_corr"] = 0.0
    lead1_values_agg["nk_morph_support"] = 0.0
    lead2_values_agg["nk_morph_support"] = 0.0
    lead1_values_agg["nk_morph_unreliability"] = 1.0
    lead2_values_agg["nk_morph_unreliability"] = 1.0
    lead1_values_agg["readability_support"] = 0.0
    lead2_values_agg["readability_support"] = 0.0
    lead1_values_agg["readability_bonus"] = 0.0
    lead2_values_agg["readability_bonus"] = 0.0
    lead1_values_agg["readability_relief_gate"] = 0.0
    lead2_values_agg["readability_relief_gate"] = 0.0
    lead1_values_agg["readability_relief_bonus"] = 0.0
    lead2_values_agg["readability_relief_bonus"] = 0.0
    lead1_values_agg["nk_zero_cap_applied"] = 0.0
    lead2_values_agg["nk_zero_cap_applied"] = 0.0
    duration_sec = n / sampling_rate if sampling_rate > 0 else 0.0
    for lead_nk_positions, lead_num in [
        (lead1_nk_positions, 1),
        (lead2_nk_positions, 2),
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
            lead1_values_agg["peak_consistency_proxy"] = float(severity)
            lead1_values_agg["peak_consistency_count"] = float(len(lead_nk_positions))
            lead1_values_agg["readability_support"] = _compute_readability_support(
                {
                    "consistency_severity": float(severity),
                    "peak_count": float(len(lead_nk_positions)),
                    "peak_span_ratio": float(peak_info["peak_span_ratio"]),
                    "peak_density_bpm": float(peak_info["peak_density_bpm"]),
                },
                lead1_nk_quality,
                float(lead1_values_agg.get("snr", 0.0)),
            )
            morph_info = _compute_peak_morphology_unreliability(
                lead1[:n],
                lead_nk_positions,
                sampling_rate,
            )
            lead1_values_agg["nk_peak_to_mad"] = float(morph_info["peak_to_mad"])
            lead1_values_agg["nk_morph_p25_corr"] = float(morph_info["p25_corr"])
            lead1_values_agg["nk_morph_median_corr"] = float(morph_info["median_corr"])
            lead1_values_agg["nk_morph_support"] = float(morph_info["support"])
            lead1_values_agg["nk_morph_unreliability"] = float(morph_info["unreliability"])
        else:
            lead2_values_agg["nk_rr_median_sec"] = float(peak_info["rr_median_sec"])
            lead2_values_agg["nk_rr_short_ratio"] = float(peak_info["rr_short_ratio"])
            lead2_values_agg["nk_rr_duplicate_ratio"] = float(peak_info["rr_duplicate_ratio"])
            lead2_values_agg["nk_peak_density_bpm"] = float(peak_info["peak_density_bpm"])
            lead2_values_agg["nk_peak_span_ratio"] = float(peak_info["peak_span_ratio"])
            lead2_values_agg["nk_long_gap_ratio"] = float(peak_info["long_gap_ratio"])
            lead2_values_agg["nk_max_gap_sec"] = float(peak_info["max_gap_sec"])
            lead2_values_agg["peak_consistency_proxy"] = float(severity)
            lead2_values_agg["peak_consistency_count"] = float(len(lead_nk_positions))
            lead2_values_agg["readability_support"] = _compute_readability_support(
                {
                    "consistency_severity": float(severity),
                    "peak_count": float(len(lead_nk_positions)),
                    "peak_span_ratio": float(peak_info["peak_span_ratio"]),
                    "peak_density_bpm": float(peak_info["peak_density_bpm"]),
                },
                lead2_nk_quality,
                float(lead2_values_agg.get("snr", 0.0)),
            )
            morph_info = _compute_peak_morphology_unreliability(
                lead2[:n],
                lead_nk_positions,
                sampling_rate,
            )
            lead2_values_agg["nk_peak_to_mad"] = float(morph_info["peak_to_mad"])
            lead2_values_agg["nk_morph_p25_corr"] = float(morph_info["p25_corr"])
            lead2_values_agg["nk_morph_median_corr"] = float(morph_info["median_corr"])
            lead2_values_agg["nk_morph_support"] = float(morph_info["support"])
            lead2_values_agg["nk_morph_unreliability"] = float(morph_info["unreliability"])

    # Aggregate per-window scores into a final readability score per lead.
    lead_agg_cfg = config.get("lead_aggregation", {})
    q1_score, q1_window_stats = _aggregate_lead_windows(
        window_results, "lead1_score", lead_agg_cfg
    )
    q2_score, q2_window_stats = _aggregate_lead_windows(
        window_results, "lead2_score", lead_agg_cfg
    )
    # Include NK fields in aggregated values
    lead1_values_agg["nk_quality"] = lead1_nk_quality
    lead2_values_agg["nk_quality"] = lead2_nk_quality
    lead1_values_agg["nk_r_peaks_count"] = lead1_nk_r_peaks
    lead2_values_agg["nk_r_peaks_count"] = lead2_nk_r_peaks

    q1_window_score = q1_score
    q2_window_score = q2_score
    q1_flag_score = compute_quality_score(agg_flags1, config=config)
    q2_flag_score = compute_quality_score(agg_flags2, config=config)
    q1_bonus = 0.0
    q2_bonus = 0.0
    q1_relief_bonus = 0.0
    q2_relief_bonus = 0.0
    low_gain_bonus = float(lead_agg_cfg.get("low_gain_bonus", 0.20))
    low_gain_max_base = float(lead_agg_cfg.get("low_gain_max_base", 0.45))
    low_gain_max_std = float(lead_agg_cfg.get("low_gain_max_std", 0.08))
    low_gain_min_support = float(lead_agg_cfg.get("low_gain_min_support", 0.80))
    q1_peak_unreliability = _compute_peak_unreliability(lead1_values_agg)
    q2_peak_unreliability = _compute_peak_unreliability(lead2_values_agg)
    if (
        lead1_values_agg["readability_support"] >= low_gain_min_support
        and q1_window_score <= low_gain_max_base
        and q1_window_stats.get("std", 1.0) <= low_gain_max_std
        and agg_flags1.get("Bad_Electrode_Contact", 1.0) < 0.25
        and q1_peak_unreliability <= 0.15
    ):
        q1_bonus = low_gain_bonus * float(lead1_values_agg["readability_support"])
    if (
        lead2_values_agg["readability_support"] >= low_gain_min_support
        and q2_window_score <= low_gain_max_base
        and q2_window_stats.get("std", 1.0) <= low_gain_max_std
        and agg_flags2.get("Bad_Electrode_Contact", 1.0) < 0.25
        and q2_peak_unreliability <= 0.15
    ):
        q2_bonus = low_gain_bonus * float(lead2_values_agg["readability_support"])

    # Optional "readability relief" keeps low-gain but clearly readable ECG
    # from being over-penalized by pure PSD metrics. Setting max_bonus=0.0
    # restores the legacy behaviour for A/B testing.
    relief_cfg = config.get("readability_relief", {})
    if bool(relief_cfg.get("enabled", True)):
        support_threshold = float(relief_cfg.get("support_threshold", 0.45))
        support_full = float(relief_cfg.get("support_full", 0.75))
        morph_threshold = float(relief_cfg.get("morph_threshold", 0.70))
        morph_full = float(relief_cfg.get("morph_full", 0.90))
        noise_reference = max(1e-6, float(relief_cfg.get("noise_reference", 0.55)))
        low_snr_reference = max(1e-6, float(relief_cfg.get("low_snr_reference", 0.85)))
        max_relief_bonus = max(0.0, float(relief_cfg.get("max_bonus", 0.0)))

        q1_support_gate = _scaled_gate(
            float(lead1_values_agg.get("readability_support", 0.0)),
            support_threshold,
            support_full,
        )
        q2_support_gate = _scaled_gate(
            float(lead2_values_agg.get("readability_support", 0.0)),
            support_threshold,
            support_full,
        )
        q1_morph_gate = _scaled_gate(
            float(lead1_values_agg.get("nk_morph_support", 0.0)),
            morph_threshold,
            morph_full,
        )
        q2_morph_gate = _scaled_gate(
            float(lead2_values_agg.get("nk_morph_support", 0.0)),
            morph_threshold,
            morph_full,
        )
        q1_relief_gate = min(
            q1_support_gate,
            q1_morph_gate,
            float(np.clip(1.0 - q1_peak_unreliability, 0.0, 1.0)),
        )
        q2_relief_gate = min(
            q2_support_gate,
            q2_morph_gate,
            float(np.clip(1.0 - q2_peak_unreliability, 0.0, 1.0)),
        )
        q1_pressure = float(
            np.clip(
                0.5 * agg_flags1.get("Noise_Artifact", 0.0) / noise_reference
                + 0.5 * agg_flags1.get("Low_SNR", 0.0) / low_snr_reference,
                0.0,
                1.0,
            )
        )
        q2_pressure = float(
            np.clip(
                0.5 * agg_flags2.get("Noise_Artifact", 0.0) / noise_reference
                + 0.5 * agg_flags2.get("Low_SNR", 0.0) / low_snr_reference,
                0.0,
                1.0,
            )
        )
        if agg_flags1.get("Bad_Electrode_Contact", 1.0) < 0.25:
            q1_relief_bonus = max_relief_bonus * q1_relief_gate * q1_pressure
        if agg_flags2.get("Bad_Electrode_Contact", 1.0) < 0.25:
            q2_relief_bonus = max_relief_bonus * q2_relief_gate * q2_pressure
        lead1_values_agg["readability_relief_gate"] = q1_relief_gate
        lead2_values_agg["readability_relief_gate"] = q2_relief_gate

    if lead1_nk_r_peaks == 0:
        q1_window_score = min(q1_window_score, 0.25)
        q1_bonus = 0.0
        q1_relief_bonus = 0.0
        lead1_values_agg["nk_zero_cap_applied"] = 1.0
    if lead2_nk_r_peaks == 0:
        q2_window_score = min(q2_window_score, 0.25)
        q2_bonus = 0.0
        q2_relief_bonus = 0.0
        lead2_values_agg["nk_zero_cap_applied"] = 1.0

    lead1_values_agg["window_quality"] = q1_window_score
    lead2_values_agg["window_quality"] = q2_window_score
    lead1_values_agg["flag_quality"] = q1_flag_score
    lead2_values_agg["flag_quality"] = q2_flag_score
    lead1_values_agg["flag_uplift_limit"] = q1_bonus
    lead2_values_agg["flag_uplift_limit"] = q2_bonus
    lead1_values_agg["low_gain_recovery_limit"] = q1_bonus
    lead2_values_agg["low_gain_recovery_limit"] = q2_bonus
    lead1_values_agg["readability_bonus"] = q1_bonus
    lead2_values_agg["readability_bonus"] = q2_bonus
    lead1_values_agg["readability_relief_bonus"] = q1_relief_bonus
    lead2_values_agg["readability_relief_bonus"] = q2_relief_bonus

    q1_avg = max(0.0, min(1.0, q1_window_score + q1_bonus + q1_relief_bonus))
    q2_avg = max(0.0, min(1.0, q2_window_score + q2_bonus + q2_relief_bonus))
    q1_peak_penalty = 0.45 * q1_peak_unreliability
    q2_peak_penalty = 0.45 * q2_peak_unreliability
    lead1_values_agg["peak_unreliability"] = q1_peak_penalty / 0.45 if 0.45 > 0 else 0.0
    lead2_values_agg["peak_unreliability"] = q2_peak_penalty / 0.45 if 0.45 > 0 else 0.0
    q1_avg = max(0.0, q1_avg - q1_peak_penalty)
    q2_avg = max(0.0, q2_avg - q2_peak_penalty)
    q1_psd_all = q1_window_score
    q2_psd_all = q2_window_score
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
        "analysis_start_sec": analysis_start / sampling_rate if sampling_rate > 0 else 0.0,
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
