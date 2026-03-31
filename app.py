"""
ECG Sentinel Quality - Streamlit GUI
Minimal testing environment for Sentinel .dat file analysis.

Processes .dat files using ported C# logic from Stream-Analysis-Service
and quality analysis from Ecg-Interpretation-Python-Service.
"""

import glob
import json
import os
import re
import warnings

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3
from plotly.subplots import make_subplots
from src.dat_parser import are_files_consecutive, parse_dat_file
from src.filters import preprocess_dat_signal
from src.optimizer import (
    GRADE_TARGETS,
    decode_vector,
    evaluate_config,
    run_optimization,
)
from src.qrs_detector import compute_heart_rate, detect_qrs
from src.quality import analyze_holter_quality, load_all_presets, save_preset


@st.cache_data
def _cached_load_presets():
    """Cached wrapper around load_all_presets to avoid repeated disk I/O."""
    return load_all_presets()


# Single source of truth for flag names and threshold UI config
_THRESH_CONFIG = {
    "Muscle_Artifact": {"min": 0.0, "max": 1.0, "step": 0.005, "format": "%.3f"},
    "Bad_Electrode_Contact": {"min": 0.0, "max": 2000.0, "step": 5.0, "format": "%.0f"},
    "Powerline_Interference": {"min": 0.0, "max": 1.0, "step": 0.005, "format": "%.3f"},
    "Periodic_Artifact": {"min": 0.0, "max": 1.0, "step": 0.005, "format": "%.3f"},
    "Baseline_Drift": {"min": 0.0, "max": 1.0, "step": 0.01, "format": "%.2f"},
    "Low_SNR": {"min": 0.0, "max": 50.0, "step": 1.0, "format": "%.0f"},
}
_FLAG_NAMES = list(_THRESH_CONFIG.keys())
_ALL_FLAG_NAMES = ["Noise_Artifact", "Bad_Electrode_Contact", "Baseline_Drift", "Low_SNR"]
_PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# -- Page config ---------------------------------------------------------------

st.set_page_config(
    page_title="ECG Sentinel Quality",
    page_icon="\u2764",
    layout="wide",
)

# Reduce dead space above sidebar tabs
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        padding-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ECG Sentinel Quality Analyzer")
st.caption("Testing environment for Sentinel Holter .dat files")

# -- Sidebar: file selection & parameters --------------------------------------

# -- Load presets for config tab -----------------------------------------------

_presets_data = _cached_load_presets()
_preset_names = list(_presets_data.get("presets", {}).keys())
_default_preset = _presets_data.get(
    "default", _preset_names[0] if _preset_names else "hardcoded_default"
)

# Initialize session state for config if not present
if "cfg_preset" not in st.session_state:
    st.session_state["cfg_preset"] = _default_preset


def _populate_config_state(preset_name: str) -> None:
    """Populate all cfg_* session state keys from a preset.
    Uses _FLAG_NAMES to ensure every known flag gets a value even if preset omits it."""
    from src.quality import _FALLBACK_CONFIG

    presets_data = _cached_load_presets()
    preset = presets_data.get("presets", {}).get(preset_name, {})
    if not preset:
        return
    fallback_thresholds = _FALLBACK_CONFIG["thresholds"]
    fallback_weights = _FALLBACK_CONFIG["flags_weights"]
    # Thresholds — fill all known flags with fallback for any missing
    thresholds = preset.get("thresholds", {})
    for flag_name in _FLAG_NAMES:
        bounds = thresholds.get(
            flag_name, fallback_thresholds.get(flag_name, (0.0, 1.0))
        )
        st.session_state[f"cfg_thresh_{flag_name}_low"] = float(bounds[0])
        st.session_state[f"cfg_thresh_{flag_name}_high"] = float(bounds[1])
    # Weights — fill all known flags, including NK peak consistency
    weights = preset.get("flags_weights", {})
    for flag_name in _ALL_FLAG_NAMES:
        st.session_state[f"cfg_weight_{flag_name}"] = float(
            weights.get(flag_name, fallback_weights.get(flag_name, 0.2))
        )
    # Grade thresholds
    grades = preset.get("grade_thresholds", {})
    st.session_state["cfg_grade_good"] = float(grades.get("good", 0.85))
    st.session_state["cfg_grade_questionable"] = float(grades.get("questionable", 0.65))
    # NeuroKit
    nk = preset.get("neurokit", {})
    st.session_state["cfg_nk_enabled"] = bool(nk.get("enabled", False))
    st.session_state["cfg_nk_method"] = nk.get("method", "averageQRS")
    st.session_state["cfg_nk_weight"] = float(nk.get("weight", 0.0))
    relief = preset.get("readability_relief", {})
    st.session_state["cfg_relief_enabled"] = bool(relief.get("enabled", True))
    st.session_state["cfg_relief_support_threshold"] = float(
        relief.get("support_threshold", 0.45)
    )
    st.session_state["cfg_relief_support_full"] = float(
        relief.get("support_full", 0.75)
    )
    st.session_state["cfg_relief_morph_threshold"] = float(
        relief.get("morph_threshold", 0.70)
    )
    st.session_state["cfg_relief_morph_full"] = float(
        relief.get("morph_full", 0.90)
    )
    st.session_state["cfg_relief_noise_reference"] = float(
        relief.get("noise_reference", 0.55)
    )
    st.session_state["cfg_relief_low_snr_reference"] = float(
        relief.get("low_snr_reference", 0.85)
    )
    st.session_state["cfg_relief_max_bonus"] = float(relief.get("max_bonus", 0.18))


# Handle pending preset switch (set before rerun from save-as-new flow)
if "_pending_preset" in st.session_state:
    _switch_to = st.session_state.pop("_pending_preset")
    st.session_state["cfg_preset"] = _switch_to
    _populate_config_state(_switch_to)
# Populate defaults on first run (or when new config keys are missing)
elif "cfg_thresh_Muscle_Artifact_low" not in st.session_state:
    _populate_config_state(st.session_state["cfg_preset"])
# Backfill config keys added after initial session was created
else:
    _preset_name = st.session_state.get("cfg_preset", _default_preset)
    _preset = _cached_load_presets().get("presets", {}).get(_preset_name, {})
    _fallback_thresholds = {
        "Muscle_Artifact": (0.045, 0.10),
        "Bad_Electrode_Contact": (10.0, 800.0),
        "Powerline_Interference": (0.01, 0.05),
        "Periodic_Artifact": (0.10, 0.45),
        "Baseline_Drift": (0.03, 0.90),
        "Low_SNR": (22.0, 12.0),
    }
    _fallback_weights = {
        "Noise_Artifact": 0.35,
        "Bad_Electrode_Contact": 0.15,
        "Baseline_Drift": 0.10,
        "Low_SNR": 0.40,
    }
    _preset_thresholds = _preset.get("thresholds", {})
    _preset_weights = _preset.get("flags_weights", {})
    _preset_grades = _preset.get("grade_thresholds", {})
    _preset_nk = _preset.get("neurokit", {})

    for flag_name in _FLAG_NAMES:
        low_key = f"cfg_thresh_{flag_name}_low"
        high_key = f"cfg_thresh_{flag_name}_high"
        bounds = _preset_thresholds.get(flag_name, _fallback_thresholds.get(flag_name, (0.0, 1.0)))
        if low_key not in st.session_state:
            st.session_state[low_key] = float(bounds[0])
        if high_key not in st.session_state:
            st.session_state[high_key] = float(bounds[1])

    for flag_name in _ALL_FLAG_NAMES:
        weight_key = f"cfg_weight_{flag_name}"
        if weight_key not in st.session_state:
            st.session_state[weight_key] = float(
                _preset_weights.get(flag_name, _fallback_weights.get(flag_name, 0.2))
            )

    if "cfg_grade_good" not in st.session_state:
        st.session_state["cfg_grade_good"] = float(_preset_grades.get("good", 0.85))
    if "cfg_grade_questionable" not in st.session_state:
        st.session_state["cfg_grade_questionable"] = float(
            _preset_grades.get("questionable", 0.65)
        )
    if "cfg_nk_enabled" not in st.session_state:
        st.session_state["cfg_nk_enabled"] = bool(_preset_nk.get("enabled", False))
    if "cfg_nk_method" not in st.session_state:
        st.session_state["cfg_nk_method"] = _preset_nk.get("method", "averageQRS")
    if "cfg_nk_weight" not in st.session_state:
        st.session_state["cfg_nk_weight"] = float(_preset_nk.get("weight", 0.0))
    _preset_relief = _preset.get("readability_relief", {})
    if "cfg_relief_enabled" not in st.session_state:
        st.session_state["cfg_relief_enabled"] = bool(_preset_relief.get("enabled", True))
    if "cfg_relief_support_threshold" not in st.session_state:
        st.session_state["cfg_relief_support_threshold"] = float(
            _preset_relief.get("support_threshold", 0.45)
        )
    if "cfg_relief_support_full" not in st.session_state:
        st.session_state["cfg_relief_support_full"] = float(
            _preset_relief.get("support_full", 0.75)
        )
    if "cfg_relief_morph_threshold" not in st.session_state:
        st.session_state["cfg_relief_morph_threshold"] = float(
            _preset_relief.get("morph_threshold", 0.70)
        )
    if "cfg_relief_morph_full" not in st.session_state:
        st.session_state["cfg_relief_morph_full"] = float(
            _preset_relief.get("morph_full", 0.90)
        )
    if "cfg_relief_noise_reference" not in st.session_state:
        st.session_state["cfg_relief_noise_reference"] = float(
            _preset_relief.get("noise_reference", 0.55)
        )
    if "cfg_relief_low_snr_reference" not in st.session_state:
        st.session_state["cfg_relief_low_snr_reference"] = float(
            _preset_relief.get("low_snr_reference", 0.85)
        )
    if "cfg_relief_max_bonus" not in st.session_state:
        st.session_state["cfg_relief_max_bonus"] = float(
            _preset_relief.get("max_bonus", 0.18)
        )


def _on_preset_change() -> None:
    """Callback when preset selectbox changes."""
    _populate_config_state(st.session_state["cfg_preset"])


def _build_quality_config() -> dict:
    """Build quality config dict from current session state sliders."""
    return {
        "_preset_name": st.session_state.get("cfg_preset", "custom"),
        "thresholds": {
            flag: (
                st.session_state.get(f"cfg_thresh_{flag}_low", 0.0),
                st.session_state.get(f"cfg_thresh_{flag}_high", 0.0),
            )
            for flag in _FLAG_NAMES
        },
        "flags_weights": {
            flag: st.session_state.get(f"cfg_weight_{flag}", 0.2)
            for flag in _ALL_FLAG_NAMES
        },
        "neurokit": {
            "enabled": st.session_state.get("cfg_nk_enabled", False),
            "method": st.session_state.get("cfg_nk_method", "averageQRS"),
            "weight": st.session_state.get("cfg_nk_weight", 0.0),
        },
        "readability_relief": {
            "enabled": st.session_state.get("cfg_relief_enabled", True),
            "support_threshold": st.session_state.get("cfg_relief_support_threshold", 0.45),
            "support_full": st.session_state.get("cfg_relief_support_full", 0.75),
            "morph_threshold": st.session_state.get("cfg_relief_morph_threshold", 0.70),
            "morph_full": st.session_state.get("cfg_relief_morph_full", 0.90),
            "noise_reference": st.session_state.get("cfg_relief_noise_reference", 0.55),
            "low_snr_reference": st.session_state.get("cfg_relief_low_snr_reference", 0.85),
            "max_bonus": st.session_state.get("cfg_relief_max_bonus", 0.18),
        },
        "grade_thresholds": {
            "good": st.session_state.get("cfg_grade_good", 0.85),
            "questionable": st.session_state.get("cfg_grade_questionable", 0.65),
        },
    }


def _apply_optimized_config(config: dict) -> None:
    """Write optimizer result into cfg_* session state keys."""
    thresholds = config.get("thresholds", {})
    for flag_name in _FLAG_NAMES:
        bounds = thresholds.get(flag_name, (0.0, 1.0))
        st.session_state[f"cfg_thresh_{flag_name}_low"] = float(bounds[0])
        st.session_state[f"cfg_thresh_{flag_name}_high"] = float(bounds[1])
    weights = config.get("flags_weights", {})
    for flag_name in _ALL_FLAG_NAMES:
        st.session_state[f"cfg_weight_{flag_name}"] = float(weights.get(flag_name, 0.2))
    grades = config.get("grade_thresholds", {})
    st.session_state["cfg_grade_good"] = float(grades.get("good", 0.85))
    st.session_state["cfg_grade_questionable"] = float(grades.get("questionable", 0.65))
    nk = config.get("neurokit", {})
    st.session_state["cfg_nk_enabled"] = bool(nk.get("enabled", False))
    st.session_state["cfg_nk_method"] = nk.get("method", "averageQRS")
    st.session_state["cfg_nk_weight"] = float(nk.get("weight", 0.0))
    relief = config.get("readability_relief", {})
    st.session_state["cfg_relief_enabled"] = bool(relief.get("enabled", True))
    st.session_state["cfg_relief_support_threshold"] = float(relief.get("support_threshold", 0.45))
    st.session_state["cfg_relief_support_full"] = float(relief.get("support_full", 0.75))
    st.session_state["cfg_relief_morph_threshold"] = float(relief.get("morph_threshold", 0.70))
    st.session_state["cfg_relief_morph_full"] = float(relief.get("morph_full", 0.90))
    st.session_state["cfg_relief_noise_reference"] = float(relief.get("noise_reference", 0.55))
    st.session_state["cfg_relief_low_snr_reference"] = float(
        relief.get("low_snr_reference", 0.85)
    )
    st.session_state["cfg_relief_max_bonus"] = float(relief.get("max_bonus", 0.18))


def _guess_grade(folder_name: str) -> str:
    """Heuristic default grade from folder name keywords."""
    name = folder_name.lower()
    # Keywords suggesting bad quality
    bad_keywords = [
        "no signal",
        "no electrode",
        "line no",
        "flat",
        "disconnect",
        "no contact",
        "dead",
    ]
    for kw in bad_keywords:
        if kw in name:
            return "Not usable"
    # Keywords suggesting questionable quality
    quest_keywords = [
        "noise",
        "artefact",
        "artifact",
        "drift",
        "interference",
        "nooise",
        "spike",
        "pause",
        "small noise",
    ]
    for kw in quest_keywords:
        if kw in name:
            return "Questionable"
    # Keywords suggesting good quality
    good_keywords = ["normal", "clean", "good", "rest"]
    for kw in good_keywords:
        if kw in name:
            return "Good"
    return "Questionable"


# -- Sidebar: tabs for Input & Config -----------------------------------------

with st.sidebar:
    sidebar_input_tab, sidebar_config_tab, sidebar_optimize_tab = st.tabs(
        ["Input", "Config", "Optimize"]
    )

    # ---- Input Tab ----
    with sidebar_input_tab:

        input_mode = st.radio(
            "Source", ["Select folder", "Upload files"], horizontal=True
        )

        dat_files_data = {}

        if input_mode == "Upload files":
            uploaded = st.file_uploader(
                "Upload .dat files",
                type=["dat"],
                accept_multiple_files=True,
            )
            if uploaded:
                for f in uploaded:
                    dat_files_data[f.name] = f.read()
        else:
            base_folder = st.text_input("Base folder", value="data")
            if base_folder and os.path.isdir(base_folder):
                subdirs = sorted(
                    [
                        d
                        for d in os.listdir(base_folder)
                        if os.path.isdir(os.path.join(base_folder, d))
                    ]
                )
                top_dats = sorted(glob.glob(os.path.join(base_folder, "*.dat")))

                options = []
                if top_dats:
                    options.append(f". ({len(top_dats)} files)")
                for sd in subdirs:
                    count = len(glob.glob(os.path.join(base_folder, sd, "*.dat")))
                    if count > 0:
                        options.append(f"{sd} ({count} files)")

                if options:
                    selected = st.selectbox("Subfolder", options)
                    folder_name = selected.split(" (")[0]
                    if folder_name == ".":
                        folder = base_folder
                    else:
                        folder = os.path.join(base_folder, folder_name)

                    found = sorted(glob.glob(os.path.join(folder, "*.dat")))
                    st.write(f"Found {len(found)} .dat files")
                    for fp in found:
                        with open(fp, "rb") as fh:
                            dat_files_data[os.path.basename(fp)] = fh.read()
                else:
                    st.warning("No .dat files found in folder or subfolders")
            elif base_folder:
                st.warning("Folder not found")

        st.divider()
        st.header("Parameters")

        notch_freq = st.multiselect(
            "Notch filter frequencies (Hz)",
            options=[50, 60, 100],
            default=[50, 100],
        )

        process_btn = st.button("Analyze", type="primary", width="stretch")

        st.divider()
        st.header("API Validation")

        api_base_url = st.text_input(
            "API Base URL",
            value=st.session_state.get("api_base_url", "https://devgateway.commwellmedical.com"),
            key="api_base_url",
        )
        api_bearer_token = st.text_input(
            "Bearer Token",
            type="password",
            key="api_bearer_token",
        )
        validate_api_btn = st.button("Validate via API", width="stretch")

    # ---- Config Tab ----
    with sidebar_config_tab:

        # Reload presets to pick up any saves during this session
        _live_presets = _cached_load_presets()
        _live_names = list(_live_presets.get("presets", {}).keys())

        st.selectbox(
            "Preset",
            options=_live_names,
            key="cfg_preset",
            on_change=_on_preset_change,
        )

        # -- Thresholds --
        st.subheader("Thresholds")

        for flag_name, cfg in _THRESH_CONFIG.items():
            label = flag_name.replace("_", " ")
            if flag_name == "Low_SNR":
                label_low = f"{label} - Good SNR (high)"
                label_high = f"{label} - Bad SNR (low)"
            else:
                label_low = f"{label} - Low"
                label_high = f"{label} - High"
            st.number_input(
                label_low,
                min_value=cfg["min"],
                max_value=cfg["max"],
                step=cfg["step"],
                format=cfg["format"],
                key=f"cfg_thresh_{flag_name}_low",
            )
            st.number_input(
                label_high,
                min_value=cfg["min"],
                max_value=cfg["max"],
                step=cfg["step"],
                format=cfg["format"],
                key=f"cfg_thresh_{flag_name}_high",
            )

        # -- Flag Weights --
        st.subheader("Flag Weights")

        for flag_name in _ALL_FLAG_NAMES:
            st.slider(
                flag_name.replace("_", " "),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key=f"cfg_weight_{flag_name}",
            )
        weight_sum = sum(
            st.session_state.get(f"cfg_weight_{f}", 0.2) for f in _ALL_FLAG_NAMES
        )
        st.caption(f"Weight sum: {weight_sum:.2f}")
        if abs(weight_sum - 1.0) > 0.05:
            st.warning(
                "Weights should sum to ~1.0 for properly calibrated grade thresholds."
            )

        # -- Grade Thresholds --
        st.subheader("Grade Thresholds")

        st.slider(
            "Good (above)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_grade_good",
        )
        st.slider(
            "Questionable (above)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_grade_questionable",
        )

        # -- NeuroKit2 --
        st.subheader("NeuroKit2")

        st.toggle("Enable NeuroKit2", key="cfg_nk_enabled")
        nk_disabled = not st.session_state.get("cfg_nk_enabled", False)
        st.selectbox(
            "Method",
            options=["averageQRS", "zhao2018", "orphanidou2015"],
            key="cfg_nk_method",
            disabled=nk_disabled,
        )
        st.slider(
            "NK Weight",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="cfg_nk_weight",
            disabled=nk_disabled,
        )

        # -- Readability Relief --
        st.subheader("Readability Relief")
        st.caption(
            "Small bonus for low-gain but clearly readable ECG. "
            "Set Max bonus to 0.0 to restore legacy behaviour."
        )
        st.toggle("Enable readability relief", key="cfg_relief_enabled")
        relief_disabled = not st.session_state.get("cfg_relief_enabled", True)
        st.slider(
            "Support threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_relief_support_threshold",
            disabled=relief_disabled,
        )
        st.slider(
            "Support full",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_relief_support_full",
            disabled=relief_disabled,
        )
        st.slider(
            "Morph threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_relief_morph_threshold",
            disabled=relief_disabled,
        )
        st.slider(
            "Morph full",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_relief_morph_full",
            disabled=relief_disabled,
        )
        st.slider(
            "Noise reference",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_relief_noise_reference",
            disabled=relief_disabled,
        )
        st.slider(
            "Low SNR reference",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_relief_low_snr_reference",
            disabled=relief_disabled,
        )
        st.slider(
            "Max relief bonus",
            min_value=0.0,
            max_value=0.5,
            step=0.01,
            key="cfg_relief_max_bonus",
            disabled=relief_disabled,
        )

        # -- Save Preset --
        st.divider()
        st.subheader("Save Preset")

        save_mode = st.radio(
            "Save mode",
            ["Overwrite current preset", "Save as new preset"],
            key="cfg_save_mode",
        )
        new_preset_name = ""
        if save_mode == "Save as new preset":
            new_preset_name = st.text_input(
                "New preset name", key="cfg_new_preset_name"
            )

        if st.button("Save Preset", width="stretch"):
            # Validate thresholds before save (F1, F7)
            _validation_errors = []
            for flag_name in _FLAG_NAMES:
                low_val = st.session_state.get(f"cfg_thresh_{flag_name}_low", 0.0)
                high_val = st.session_state.get(f"cfg_thresh_{flag_name}_high", 0.0)
                if flag_name == "Low_SNR":
                    if low_val <= high_val:
                        _validation_errors.append(
                            f"Low SNR: 'Good SNR' ({low_val}) must be > 'Bad SNR' ({high_val})"
                        )
                else:
                    if low_val > high_val:
                        _validation_errors.append(
                            f"{flag_name}: Low ({low_val}) must be <= High ({high_val})"
                        )
            # Validate grade thresholds (F3)
            _good = st.session_state.get("cfg_grade_good", 0.85)
            _quest = st.session_state.get("cfg_grade_questionable", 0.65)
            if _good <= _quest:
                _validation_errors.append(
                    f"Grade 'Good' ({_good}) must be > 'Questionable' ({_quest})"
                )
            _relief_support_threshold = st.session_state.get("cfg_relief_support_threshold", 0.45)
            _relief_support_full = st.session_state.get("cfg_relief_support_full", 0.75)
            if _relief_support_full < _relief_support_threshold:
                _validation_errors.append(
                    "Readability relief: Support full must be >= Support threshold"
                )
            _relief_morph_threshold = st.session_state.get("cfg_relief_morph_threshold", 0.70)
            _relief_morph_full = st.session_state.get("cfg_relief_morph_full", 0.90)
            if _relief_morph_full < _relief_morph_threshold:
                _validation_errors.append(
                    "Readability relief: Morph full must be >= Morph threshold"
                )

            if _validation_errors:
                for err in _validation_errors:
                    st.error(err)
            else:
                # Build preset dict from session state
                _save_preset = {
                    "thresholds": {},
                    "flags_weights": {},
                    "neurokit": {
                        "enabled": st.session_state.get("cfg_nk_enabled", False),
                        "method": st.session_state.get("cfg_nk_method", "averageQRS"),
                        "weight": st.session_state.get("cfg_nk_weight", 0.0),
                    },
                    "readability_relief": {
                        "enabled": st.session_state.get("cfg_relief_enabled", True),
                        "support_threshold": st.session_state.get(
                            "cfg_relief_support_threshold", 0.45
                        ),
                        "support_full": st.session_state.get(
                            "cfg_relief_support_full", 0.75
                        ),
                        "morph_threshold": st.session_state.get(
                            "cfg_relief_morph_threshold", 0.70
                        ),
                        "morph_full": st.session_state.get(
                            "cfg_relief_morph_full", 0.90
                        ),
                        "noise_reference": st.session_state.get(
                            "cfg_relief_noise_reference", 0.55
                        ),
                        "low_snr_reference": st.session_state.get(
                            "cfg_relief_low_snr_reference", 0.85
                        ),
                        "max_bonus": st.session_state.get("cfg_relief_max_bonus", 0.18),
                    },
                    "grade_thresholds": {
                        "good": _good,
                        "questionable": _quest,
                    },
                }
                for flag_name in _FLAG_NAMES:
                    _save_preset["thresholds"][flag_name] = [
                        st.session_state.get(f"cfg_thresh_{flag_name}_low", 0.0),
                        st.session_state.get(f"cfg_thresh_{flag_name}_high", 0.0),
                    ]
                for flag_name in _ALL_FLAG_NAMES:
                    _save_preset["flags_weights"][flag_name] = st.session_state.get(
                        f"cfg_weight_{flag_name}", 0.2
                    )

                # Preserve _threshold_docs from current preset
                _presets_data_current = _cached_load_presets()
                target_name = st.session_state.get("cfg_preset", "holter_200hz")

                if save_mode == "Save as new preset":
                    if not new_preset_name or not new_preset_name.strip():
                        st.error("Please enter a preset name.")
                    elif not _PRESET_NAME_RE.match(new_preset_name.strip()):
                        st.error(
                            "Preset name must be 1-64 characters: letters, digits, hyphens, underscores only."
                        )
                    else:
                        target_name = new_preset_name.strip()
                        # Carry over _threshold_docs from current preset if available
                        current_docs = (
                            _presets_data_current.get("presets", {})
                            .get(st.session_state.get("cfg_preset", ""), {})
                            .get("_threshold_docs")
                        )
                        if current_docs:
                            _save_preset["_threshold_docs"] = current_docs
                        current_baseline_docs = (
                            _presets_data_current.get("presets", {})
                            .get(st.session_state.get("cfg_preset", ""), {})
                            .get("_baseline_visibility_docs")
                        )
                        if current_baseline_docs:
                            _save_preset["_baseline_visibility_docs"] = current_baseline_docs
                        current_relief_docs = (
                            _presets_data_current.get("presets", {})
                            .get(st.session_state.get("cfg_preset", ""), {})
                            .get("_readability_relief_docs")
                        )
                        if current_relief_docs:
                            _save_preset["_readability_relief_docs"] = current_relief_docs
                        save_preset(target_name, _save_preset)
                        _cached_load_presets.clear()
                        st.session_state["_pending_preset"] = target_name
                        st.rerun()
                else:
                    # Preserve _threshold_docs from existing preset
                    existing = _presets_data_current.get("presets", {}).get(
                        target_name, {}
                    )
                    if "_threshold_docs" in existing:
                        _save_preset["_threshold_docs"] = existing["_threshold_docs"]
                    if "_baseline_visibility_docs" in existing:
                        _save_preset["_baseline_visibility_docs"] = existing[
                            "_baseline_visibility_docs"
                        ]
                    if "_readability_relief_docs" in existing:
                        _save_preset["_readability_relief_docs"] = existing[
                            "_readability_relief_docs"
                        ]
                    save_preset(target_name, _save_preset)
                    _cached_load_presets.clear()
                    st.success(f"Preset '{target_name}' saved.")

    # ---- Optimize Tab ----
    with sidebar_optimize_tab:
        st.subheader("Test Data")

        opt_base_folder = st.text_input(
            "Test data folder", value="data", key="opt_base_folder"
        )

        # Discover subfolders with .dat files
        _opt_folders = []
        if opt_base_folder and os.path.isdir(opt_base_folder):
            for d in sorted(os.listdir(opt_base_folder)):
                dp = os.path.join(opt_base_folder, d)
                if os.path.isdir(dp):
                    count = len(glob.glob(os.path.join(dp, "*.dat")))
                    if count > 0:
                        _opt_folders.append((d, count))

        if _opt_folders:
            st.caption(
                f"Found {len(_opt_folders)} folders, "
                f"{sum(c for _, c in _opt_folders)} files"
            )

            st.subheader("Expected Grades")
            _grade_options = list(GRADE_TARGETS.keys())
            _folder_grades = {}
            for folder_name, file_count in _opt_folders:
                default_grade = _guess_grade(folder_name)
                default_idx = (
                    _grade_options.index(default_grade)
                    if default_grade in _grade_options
                    else 1
                )
                grade = st.selectbox(
                    f"{folder_name} ({file_count})",
                    options=_grade_options,
                    index=default_idx,
                    key=f"opt_grade_{folder_name}",
                )
                _folder_grades[folder_name] = grade

            st.subheader("Settings")
            opt_maxiter = st.number_input(
                "Max generations",
                min_value=5,
                max_value=500,
                value=100,
                step=5,
                key="opt_maxiter",
            )
            opt_popsize = st.number_input(
                "Population size",
                min_value=5,
                max_value=100,
                value=30,
                step=5,
                key="opt_popsize",
            )
            opt_seed = st.number_input(
                "Random seed (0=random)",
                min_value=0,
                max_value=99999,
                value=42,
                key="opt_seed",
            )

            opt_notch = st.multiselect(
                "Notch freqs (Hz)",
                options=[50, 60, 100],
                default=[50, 100],
                key="opt_notch",
            )

            optimize_btn = st.button(
                "Run Optimization", type="primary", width="stretch"
            )

            if optimize_btn:
                st.session_state["_opt_folder_grades"] = _folder_grades
                st.session_state["_opt_run"] = True
        else:
            st.warning("No test folders with .dat files found")
            optimize_btn = False

# -- Optimization execution (main area) ----------------------------------------

if st.session_state.get("_opt_run"):
    st.session_state["_opt_run"] = False
    _opt_fg = st.session_state.get("_opt_folder_grades", {})
    _opt_bf = st.session_state.get("opt_base_folder", "data")
    _opt_notch = st.session_state.get("opt_notch", [50, 100])
    _opt_maxiter = st.session_state.get("opt_maxiter", 100)
    _opt_popsize = st.session_state.get("opt_popsize", 30)
    _opt_seed_val = st.session_state.get("opt_seed", 42)
    _opt_seed = _opt_seed_val if _opt_seed_val > 0 else None

    _opt_progress = st.progress(0, text="Starting optimization...")

    def _update_progress(stage: str, frac: float):
        _opt_progress.progress(min(frac, 1.0), text=stage)

    try:
        opt_result = run_optimization(
            base_folder=_opt_bf,
            folder_grades=_opt_fg,
            notch_freqs=_opt_notch,
            current_config=_build_quality_config(),
            thresh_config=_THRESH_CONFIG,
            maxiter=_opt_maxiter,
            popsize=_opt_popsize,
            seed=_opt_seed,
            progress_callback=_update_progress,
        )
        _opt_progress.empty()
        st.session_state["_opt_result"] = opt_result
    except Exception as exc:
        _opt_progress.empty()
        st.error(f"Optimization failed: {exc}")

if "_opt_result" in st.session_state:
    opt_result = st.session_state["_opt_result"]

    st.subheader("Optimization Results")

    # Metrics row
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Best Loss", f"{opt_result.best_loss:.4f}")
    with mc2:
        st.metric("Evaluations", f"{opt_result.n_evaluations:,}")
    with mc3:
        elapsed_m = opt_result.elapsed_seconds / 60
        st.metric("Time", f"{elapsed_m:.1f} min")

    # Before / After comparison table
    st.subheader("Before vs After")
    _comp_rows = []
    for before, after in zip(opt_result.before_results, opt_result.after_results):
        _comp_rows.append(
            {
                "Folder": before["folder"],
                "Expected": before["expected_grade"],
                "Before Score": f"{before['avg_score']:.3f}",
                "Before Grade": before["actual_grade"],
                "After Score": f"{after['avg_score']:.3f}",
                "After Grade": after["actual_grade"],
                "Match": "Yes" if after["match"] else "No",
            }
        )
    st.table(_comp_rows)

    # Match summary
    _matches_before = sum(1 for r in opt_result.before_results if r["match"])
    _matches_after = sum(1 for r in opt_result.after_results if r["match"])
    _total = len(opt_result.after_results)
    st.caption(
        f"Grade matches: {_matches_before}/{_total} before -> "
        f"{_matches_after}/{_total} after"
    )

    # Convergence chart
    if opt_result.convergence_history:
        conv_fig = go.Figure()
        conv_fig.add_trace(
            go.Scatter(
                y=opt_result.convergence_history,
                mode="lines+markers",
                line=dict(color="#00ffff"),
                name="Best loss",
            )
        )
        conv_fig.update_layout(
            title="Convergence (best loss per generation)",
            xaxis_title="Generation",
            yaxis_title="Loss",
            height=300,
            template="plotly_dark",
        )
        st.plotly_chart(conv_fig, use_container_width=True)

    # Optimized parameters (expandable)
    with st.expander("Optimized Parameters"):
        best_cfg = opt_result.best_config
        st.markdown("**Thresholds**")
        for flag, bounds in best_cfg["thresholds"].items():
            st.caption(f"{flag}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
        st.markdown("**Weights**")
        for flag, w in best_cfg["flags_weights"].items():
            st.caption(f"{flag}: {w:.4f}")
        w_sum = sum(best_cfg["flags_weights"].values())
        st.caption(f"Weight sum: {w_sum:.4f}")
        st.markdown("**Grade Thresholds**")
        st.caption(
            f"Good: {best_cfg['grade_thresholds']['good']:.4f}, "
            f"Questionable: {best_cfg['grade_thresholds']['questionable']:.4f}"
        )
        st.markdown("**NeuroKit**")
        nk = best_cfg["neurokit"]
        st.caption(
            f"Enabled: {nk['enabled']}, "
            f"Method: {nk['method']}, "
            f"Weight: {nk['weight']:.4f}"
        )

    # Action buttons
    acol1, acol2 = st.columns(2)
    with acol1:
        if st.button("Apply to Config", type="primary"):
            _apply_optimized_config(opt_result.best_config)
            st.success("Config updated! Switch to Config tab to see values.")
    with acol2:
        save_name = st.text_input("Preset name", value="optimized", key="opt_save_name")
        if st.button("Save as Preset"):
            if save_name and _PRESET_NAME_RE.match(save_name.strip()):
                save_preset(save_name.strip(), opt_result.best_config)
                _cached_load_presets.clear()
                st.success(f"Saved preset '{save_name.strip()}'")
            else:
                st.error(
                    "Invalid name (1-64 chars: letters, digits, hyphens, underscores)"
                )

# -- Processing ----------------------------------------------------------------

if not dat_files_data:
    st.info("Upload or select .dat files to begin analysis.")
    st.stop()

if process_btn:
    # Validate config before analysis (F1, F3, F7)
    _analysis_errors = []
    for flag in _FLAG_NAMES:
        low_v = st.session_state.get(f"cfg_thresh_{flag}_low", 0.0)
        high_v = st.session_state.get(f"cfg_thresh_{flag}_high", 0.0)
        if flag == "Low_SNR":
            if low_v <= high_v:
                _analysis_errors.append(
                    f"Low SNR: 'Good SNR' ({low_v}) must be > 'Bad SNR' ({high_v})"
                )
        else:
            if low_v > high_v:
                _analysis_errors.append(
                    f"{flag}: Low ({low_v}) must be <= High ({high_v})"
                )
    _g = st.session_state.get("cfg_grade_good", 0.85)
    _q = st.session_state.get("cfg_grade_questionable", 0.65)
    if _g <= _q:
        _analysis_errors.append(f"Grade 'Good' ({_g}) must be > 'Questionable' ({_q})")
    _relief_support_threshold = st.session_state.get("cfg_relief_support_threshold", 0.45)
    _relief_support_full = st.session_state.get("cfg_relief_support_full", 0.75)
    if _relief_support_full < _relief_support_threshold:
        _analysis_errors.append(
            "Readability relief: Support full must be >= Support threshold"
        )
    _relief_morph_threshold = st.session_state.get("cfg_relief_morph_threshold", 0.70)
    _relief_morph_full = st.session_state.get("cfg_relief_morph_full", 0.90)
    if _relief_morph_full < _relief_morph_threshold:
        _analysis_errors.append(
            "Readability relief: Morph full must be >= Morph threshold"
        )
    if _analysis_errors:
        for ae in _analysis_errors:
            st.error(ae)
        st.stop()

    # Build quality config dict from session state sliders
    quality_config = _build_quality_config()

    # Sort files by name for consistent ordering
    sorted_files = sorted(dat_files_data.keys())
    st.write(f"Processing **{len(sorted_files)}** files: {', '.join(sorted_files)}")

    # Check consecutiveness
    if len(sorted_files) >= 2:
        file_bytes = [dat_files_data[fn] for fn in sorted_files]
        consecutive = are_files_consecutive(file_bytes)
        if consecutive:
            st.success("Files are consecutive (packet numbers follow sequentially)")
        else:
            st.warning("Files are NOT consecutive - gaps detected between packets")

    # Process each file
    all_results = {}
    progress = st.progress(0, text="Parsing files...")

    for idx, fname in enumerate(sorted_files):
        progress.progress((idx + 1) / len(sorted_files), text=f"Processing {fname}...")
        try:
            raw_data = dat_files_data[fname]
            lead1_raw, lead2_raw = parse_dat_file(raw_data)

            # Preprocess (HP filter + resample 125->200 Hz + notch)
            lead1 = preprocess_dat_signal(lead1_raw, notch_freqs=notch_freq)
            lead2 = preprocess_dat_signal(lead2_raw, notch_freqs=notch_freq)

            # QRS detection on both leads
            markers1 = detect_qrs(lead1, fs=200)
            markers2 = detect_qrs(lead2, fs=200)

            # Heart rate stats
            hr1 = compute_heart_rate(markers1, fs=200)
            hr2 = compute_heart_rate(markers2, fs=200)

            # Pick best lead (lower HR std = more reliable, matching C# decide())
            if hr1["hr_std"] <= hr2["hr_std"] and hr1["hr_mean"] > 0:
                best_lead_idx = 1
            else:
                best_lead_idx = 2

            # Quality analysis with config from sidebar
            quality = analyze_holter_quality(
                lead1, lead2, sampling_rate=200, config=quality_config,
                markers1=markers1, markers2=markers2,
            )

            all_results[fname] = {
                "lead1_raw": lead1_raw,
                "lead2_raw": lead2_raw,
                "lead1": lead1,
                "lead2": lead2,
                "markers1": markers1,
                "markers2": markers2,
                "hr1": hr1,
                "hr2": hr2,
                "best_lead": best_lead_idx,
                "quality": quality,
            }
        except Exception as e:
            st.error(f"Error processing {fname}: {e}")

    progress.empty()

    if all_results:
        st.session_state["results"] = all_results
        st.session_state["results_config"] = quality_config
    else:
        st.error("No files processed successfully.")
        st.stop()

if "results" not in st.session_state:
    st.info(f"{len(dat_files_data)} files loaded. Click **Analyze** to process.")
    st.stop()

# -- API Validation ------------------------------------------------------------

if validate_api_btn:
    if "results" not in st.session_state:
        st.warning("Run **Analyze** first before validating via API.")
        st.stop()
    token = st.session_state.get("api_bearer_token", "").strip()
    base_url = st.session_state.get("api_base_url", "").strip().rstrip("/")
    if not token:
        st.error("Bearer token is required for API validation.")
        st.stop()
    if not base_url:
        st.error("API Base URL is required.")
        st.stop()

    api_url = f"{base_url}/api/v1/ecg-interpretation/check-holter-quality"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    api_results = {}
    local_results = st.session_state["results"]
    progress = st.progress(0, text="Validating via API...")

    for idx, (fname, result) in enumerate(local_results.items()):
        progress.progress((idx + 1) / len(local_results), text=f"API: {fname}...")
        lead1 = result["lead1"]
        lead2 = result["lead2"]
        payload = {
            "lead1Data": lead1.tolist(),
            "lead2Data": lead2.tolist(),
            "samplingRate": 200,
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)
                resp = requests.post(
                    api_url, json=payload, headers=headers,
                    verify=False, timeout=30,
                )
            if resp.status_code == 200:
                api_results[fname] = {"success": True, "data": resp.json()}
            else:
                api_results[fname] = {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
                }
        except requests.exceptions.ConnectionError:
            api_results[fname] = {
                "success": False,
                "error": f"Connection failed to {api_url}",
            }
        except Exception as e:
            api_results[fname] = {"success": False, "error": str(e)}

    progress.empty()
    st.session_state["api_results"] = api_results

all_results = st.session_state["results"]

# -- Results display -----------------------------------------------------------

tabs = st.tabs(list(all_results.keys()))

for tab, (fname, result) in zip(tabs, all_results.items()):
    with tab:
        q = result["quality"]
        hr1 = result["hr1"]
        hr2 = result["hr2"]

        # -- Summary metrics --
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Quality Grade", q["grade"])
        with col2:
            st.metric("Overall Quality", f"{q['overall_quality']:.2f}")
        with col3:
            best = result["best_lead"]
            best_hr = hr1 if best == 1 else hr2
            st.metric(
                "HR Mean",
                f"{best_hr['hr_mean']:.0f} BPM" if best_hr["hr_mean"] > 0 else "N/A",
            )
        with col4:
            st.metric("Best Lead", f"Lead {best}")

        # -- API Validation results --
        api_results = st.session_state.get("api_results", {})
        if fname in api_results:
            api_res = api_results[fname]
            with st.expander("API Validation", expanded=True):
                if api_res["success"]:
                    api_data = api_res["data"]
                    api_score = api_data.get("overallQuality", api_data.get("overall_quality"))
                    api_grade = api_data.get("grade")

                    cmp1, cmp2, cmp3 = st.columns(3)
                    with cmp1:
                        st.markdown("**Source**")
                        st.write("Local")
                        st.write("API")
                    with cmp2:
                        st.markdown("**Quality Score**")
                        st.write(f"{q['overall_quality']:.3f}")
                        st.write(f"{api_score:.3f}" if api_score is not None else "N/A")
                    with cmp3:
                        st.markdown("**Grade**")
                        st.write(q["grade"])
                        st.write(api_grade if api_grade else "N/A")

                    if api_score is not None:
                        diff = q["overall_quality"] - api_score
                        if abs(diff) < 0.01:
                            st.success(f"Scores match (diff: {diff:+.4f})")
                        elif abs(diff) < 0.05:
                            st.warning(f"Minor difference: {diff:+.4f}")
                        else:
                            st.error(f"Significant difference: {diff:+.4f}")

                    with st.expander("Raw API Response"):
                        st.json(api_data)
                else:
                    st.error(f"API call failed: {api_res['error']}")

        # -- Quality details --
        with st.expander("Quality Details", expanded=True):
            _rc = st.session_state.get("results_config", {})
            _nk_cfg = _rc.get("neurokit", {})
            nk_enabled = _nk_cfg.get("enabled", False)
            nk_weight = _nk_cfg.get("weight", 0.0)

            qcol1, qcol2 = st.columns(2)
            for col, lead_num in [(qcol1, 1), (qcol2, 2)]:
                with col:
                    blended = q[f"lead{lead_num}_quality"]
                    psd_q = q.get(f"lead{lead_num}_psd_quality", blended)
                    nk_q = q.get(f"lead{lead_num}_nk_quality")
                    vals = q.get(f"lead{lead_num}_values", {})
                    flags = q.get(f"lead{lead_num}_flags", {})
                    is_best = q.get("quality_best_lead") == lead_num

                    header = f"Lead {lead_num}"
                    if is_best:
                        header += " (Best)"
                    st.markdown(f"#### {header}")

                    # Scores
                    st.markdown(f"**Overall Quality: {blended:.3f}**")
                    score_parts = [f"PSD (all): {psd_q:.3f}"]
                    if nk_enabled and nk_q is not None:
                        score_parts.append(f"NK: {nk_q:.3f}")
                        score_parts.append(
                            f"Weight: {1 - nk_weight:.0%}/{nk_weight:.0%}"
                        )
                    elif nk_enabled:
                        score_parts.append("NK: N/A")
                    st.caption(" | ".join(score_parts))
                    window_stats = q.get(f"lead{lead_num}_window_stats", {})
                    if window_stats:
                        st.caption(
                            "Window agg: "
                            f"mean={window_stats.get('mean', 0.0):.3f} | "
                            f"p25={window_stats.get('p25', 0.0):.3f} | "
                            f"std={window_stats.get('std', 0.0):.3f} | "
                            f"coverage={window_stats.get('coverage', 0.0):.0%}"
                        )

                    # Flags
                    st.markdown("**Flags**")
                    for flag, val in flags.items():
                        label = flag.replace("_", " ")
                        if val > 0.5:
                            st.markdown(f"- :red[{label}: {val:.3f}]")
                        elif val > 0.1:
                            st.markdown(f"- :orange[{label}: {val:.3f}]")
                        elif val > 0.0:
                            st.markdown(f"- {label}: {val:.3f}")
                        else:
                            st.markdown(f"- :green[{label}: {val:.3f}]")

                    # Raw measurement values
                    if vals:
                        st.markdown("**Measurements**")
                        _meas_labels = {
                            "snr": ("SNR", "dB", "%.1f"),
                            "qrs_amp": ("QRS Amplitude", "", "%.1f"),
                            "qrs_band_amp": ("QRS Band Amplitude", "", "%.1f"),
                            "hf_noise_rms": ("HF Noise RMS", "", "%.2f"),
                            "m_a": ("Muscle Artifact Ratio", "", "%.4f"),
                            "muscle_flag_raw": ("Raw Muscle Flag", "", "%.3f"),
                            "m_a_hf_power": ("HF Power", "", "%.4f"),
                            "m_a_broadband_hf_power": ("Broadband HF Power", "", "%.4f"),
                            "p_i": ("Powerline Interference", "", "%.4f"),
                            "powerline_flag_raw": ("Raw Powerline Flag", "", "%.3f"),
                            "p_i_db": ("Powerline Interference", "dB", "%.1f"),
                            "noise_artifact_flag": ("Noise Artifact", "", "%.3f"),
                            "transient_artifact_ratio": ("Transient Artifact Ratio", "", "%.3f"),
                            "transient_flag_raw": ("Raw Transient Flag", "", "%.3f"),
                            "periodic_artifact_ratio": ("Periodic Artifact Ratio", "", "%.4f"),
                            "periodic_artifact_freq": ("Periodic Artifact Peak", "Hz", "%.1f"),
                            "periodic_artifact_context": ("Periodic Artifact Context", "", "%.3f"),
                            "periodic_flag_raw": ("Raw Periodic Flag", "", "%.3f"),
                            "b_d": ("Baseline Drift Ratio", "", "%.4f"),
                            "baseline_psd_flag_raw": ("Raw Baseline PSD Flag", "", "%.3f"),
                            "baseline_visible_ratio": ("Visible Baseline Ratio", "", "%.3f"),
                            "baseline_visible_flag_raw": ("Raw Visible Baseline Flag", "", "%.3f"),
                            "readability_support": ("Readability Support", "", "%.3f"),
                            "readability_bonus": ("Readability Bonus", "", "%.3f"),
                            "readability_relief_gate": ("Readability Relief Gate", "", "%.3f"),
                            "readability_relief_bonus": ("Readability Relief Bonus", "", "%.3f"),
                            "window_quality": ("Window Quality Base", "", "%.3f"),
                            "flag_quality": ("Flag Quality", "", "%.3f"),
                        }
                        for key, (label, unit, fmt) in _meas_labels.items():
                            v = vals.get(key)
                            if v is not None:
                                suffix = f" {unit}" if unit else ""
                                st.caption(f"{label}: {fmt % v}{suffix}")
                        if nk_enabled:
                            nk_peaks = vals.get("nk_r_peaks_count", 0)
                            st.caption(f"NK R-peaks detected: {nk_peaks}")
                            rr_median = vals.get("nk_rr_median_sec")
                            if rr_median is not None:
                                st.caption(f"NK median RR: {rr_median:.3f} s")
                            peak_density = vals.get("nk_peak_density_bpm")
                            if peak_density is not None:
                                st.caption(f"NK peak density: {peak_density:.1f} BPM")
                            peak_span = vals.get("nk_peak_span_ratio")
                            if peak_span is not None:
                                st.caption(f"NK peak span ratio: {peak_span:.1%}")
                            rr_short_ratio = vals.get("nk_rr_short_ratio")
                            if rr_short_ratio is not None:
                                st.caption(
                                    f"NK too-short RR ratio: {rr_short_ratio:.1%}"
                                )
                            rr_dup_ratio = vals.get("nk_rr_duplicate_ratio")
                            if rr_dup_ratio is not None:
                                st.caption(
                                    f"NK duplicate-like RR ratio: {rr_dup_ratio:.1%}"
                                )
                            long_gap_ratio = vals.get("nk_long_gap_ratio")
                            if long_gap_ratio is not None:
                                st.caption(
                                    f"NK long-gap ratio: {long_gap_ratio:.1%}"
                                )
                            max_gap = vals.get("nk_max_gap_sec")
                            if max_gap is not None:
                                st.caption(f"NK max gap: {max_gap:.2f} s")
                            nk_peak_to_mad = vals.get("nk_peak_to_mad")
                            if nk_peak_to_mad is not None:
                                st.caption(f"NK peak/background ratio: {nk_peak_to_mad:.2f}")
                            nk_morph_p25 = vals.get("nk_morph_p25_corr")
                            if nk_morph_p25 is not None:
                                st.caption(f"NK morphology p25 corr: {nk_morph_p25:.3f}")
                            nk_morph_support = vals.get("nk_morph_support")
                            if nk_morph_support is not None:
                                st.caption(f"NK morphology support: {nk_morph_support:.3f}")

            # Preset & window info
            st.caption(
                f"Preset: {q.get('preset', 'N/A')} | "
                f"Analysis start: {q.get('analysis_start_sec', 0.0):.1f}s | "
                f"Window: {q.get('window_length_sec', 5.0):.1f}s / "
                f"step {q.get('window_step_sec', 1.0):.1f}s | "
                f"Quality best lead: {q.get('quality_best_lead', 'N/A')}"
            )

            # Window scores chart
            if q["window_scores"]:
                win_fig = go.Figure()
                windows = [w["window"] for w in q["window_scores"]]
                win_fig.add_trace(
                    go.Bar(
                        x=windows,
                        y=[w["lead1"] for w in q["window_scores"]],
                        name="Lead 1",
                        marker_color="#1f77b4",
                    )
                )
                win_fig.add_trace(
                    go.Bar(
                        x=windows,
                        y=[w["lead2"] for w in q["window_scores"]],
                        name="Lead 2",
                        marker_color="#ff7f0e",
                    )
                )
                win_fig.add_trace(
                    go.Scatter(
                        x=windows,
                        y=[w["overall"] for w in q["window_scores"]],
                        name="Overall",
                        mode="lines+markers",
                        line=dict(color="white", width=2),
                    )
                )
                win_fig.update_layout(
                    title=(
                        f"Quality per Window "
                        f"({q.get('window_length_sec', 5.0):.1f}s, "
                        f"step {q.get('window_step_sec', 1.0):.1f}s)"
                    ),
                    xaxis_title="Window",
                    yaxis_title="Quality Score",
                    yaxis_range=[0, 1.05],
                    barmode="group",
                    height=300,
                    template="plotly_dark",
                )
                st.plotly_chart(win_fig, width="stretch", key=f"win_chart_{fname}")

        # -- ECG Signal plots --
        with st.expander("ECG Signals", expanded=True):
            lead1 = result["lead1"]
            lead2 = result["lead2"]
            time_axis = np.arange(len(lead1)) / 200.0
            time_axis2 = np.arange(len(lead2)) / 200.0

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=(
                    "Lead 1 (filtered, 200 Hz)",
                    "Lead 2 (filtered, 200 Hz)",
                ),
                vertical_spacing=0.08,
            )

            # Lead 1
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=lead1,
                    mode="lines",
                    line=dict(color="#00ffff", width=1),
                    name="Lead 1",
                    hovertemplate="t=%{x:.3f}s<br>val=%{y:.0f}",
                ),
                row=1,
                col=1,
            )

            # Lead 2
            fig.add_trace(
                go.Scatter(
                    x=time_axis2,
                    y=lead2,
                    mode="lines",
                    line=dict(color="#00ff00", width=1),
                    name="Lead 2",
                    hovertemplate="t=%{x:.3f}s<br>val=%{y:.0f}",
                ),
                row=2,
                col=1,
            )

            # QRS markers for lead 1
            m1 = result["markers1"]
            if len(m1) > 0:
                m1_pos = m1[:, 0]
                valid = m1_pos < len(lead1)
                m1_times = m1_pos[valid] / 200.0
                m1_vals = lead1[m1_pos[valid]]
                fig.add_trace(
                    go.Scatter(
                        x=m1_times,
                        y=m1_vals,
                        mode="markers",
                        marker=dict(color="yellow", size=6, symbol="triangle-down"),
                        name="QRS L1",
                        hovertemplate="Beat @%{x:.2f}s",
                    ),
                    row=1,
                    col=1,
                )

            # QRS markers for lead 2
            m2 = result["markers2"]
            if len(m2) > 0:
                m2_pos = m2[:, 0]
                valid = m2_pos < len(lead2)
                m2_times = m2_pos[valid] / 200.0
                m2_vals = lead2[m2_pos[valid]]
                fig.add_trace(
                    go.Scatter(
                        x=m2_times,
                        y=m2_vals,
                        mode="markers",
                        marker=dict(color="yellow", size=6, symbol="triangle-down"),
                        name="QRS L2",
                        hovertemplate="Beat @%{x:.2f}s",
                    ),
                    row=2,
                    col=1,
                )

            fig.update_layout(
                height=500,
                template="plotly_dark",
                xaxis2_title="Time (seconds)",
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
            )
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)

            st.plotly_chart(fig, width="stretch", key=f"ecg_chart_{fname}")

        # -- Raw vs Filtered comparison --
        raw_open_key = f"raw_open_{fname}"
        with st.expander(
            "Raw vs Filtered Comparison",
            expanded=st.session_state.get(raw_open_key, False),
        ):
            lead_choice = st.radio(
                "Lead",
                [1, 2],
                horizontal=True,
                key=f"raw_{fname}",
                on_change=lambda k=raw_open_key: st.session_state.__setitem__(k, True),
            )
            raw = result["lead1_raw"] if lead_choice == 1 else result["lead2_raw"]
            filt = result["lead1"] if lead_choice == 1 else result["lead2"]

            raw_time = np.arange(len(raw)) / 125.0
            filt_time = np.arange(len(filt)) / 200.0

            comp_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Raw (125 Hz)", "Filtered (200 Hz)"),
            )
            comp_fig.add_trace(
                go.Scatter(
                    x=raw_time,
                    y=raw,
                    mode="lines",
                    line=dict(color="gray", width=1),
                    name="Raw",
                ),
                row=1,
                col=1,
            )
            comp_fig.add_trace(
                go.Scatter(
                    x=filt_time,
                    y=filt,
                    mode="lines",
                    line=dict(color="#00ffff", width=1),
                    name="Filtered",
                ),
                row=2,
                col=1,
            )
            comp_fig.update_layout(
                height=400, template="plotly_dark", xaxis2_title="Time (seconds)"
            )
            st.plotly_chart(comp_fig, width="stretch", key=f"comp_chart_{fname}")

        # -- Heart Rate --
        with st.expander("Heart Rate Analysis"):
            hrcol1, hrcol2 = st.columns(2)
            with hrcol1:
                st.write("**Lead 1**")
                h1 = result["hr1"]
                if h1["hr_mean"] > 0:
                    st.write(f"Mean HR: {h1['hr_mean']:.1f} BPM")
                    st.write(f"HR Std: {h1['hr_std']:.1f}")
                    st.write(f"HR Range: {h1['hr_min']:.0f} - {h1['hr_max']:.0f} BPM")
                    st.write(f"Beats detected: {len(result['markers1'])}")
                else:
                    st.write("No beats detected")
            with hrcol2:
                st.write("**Lead 2**")
                h2 = result["hr2"]
                if h2["hr_mean"] > 0:
                    st.write(f"Mean HR: {h2['hr_mean']:.1f} BPM")
                    st.write(f"HR Std: {h2['hr_std']:.1f}")
                    st.write(f"HR Range: {h2['hr_min']:.0f} - {h2['hr_max']:.0f} BPM")
                    st.write(f"Beats detected: {len(result['markers2'])}")
                else:
                    st.write("No beats detected")

            # RR interval plot
            best_hr = result["hr1"] if result["best_lead"] == 1 else result["hr2"]
            rr = best_hr["rr_intervals"]
            if rr:
                rr_fig = go.Figure()
                rr_fig.add_trace(
                    go.Scatter(
                        y=np.array(rr) / 200.0 * 1000,
                        mode="lines+markers",
                        line=dict(color="#00ffff"),
                        name="RR interval",
                    )
                )
                rr_fig.update_layout(
                    title="RR Intervals (Best Lead)",
                    xaxis_title="Beat #",
                    yaxis_title="RR (ms)",
                    height=250,
                    template="plotly_dark",
                )
                st.plotly_chart(rr_fig, width="stretch", key=f"rr_chart_{fname}")
