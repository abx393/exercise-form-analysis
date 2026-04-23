#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

os.environ.setdefault("MPLBACKEND", "Agg")

import FINALREP as finalrep
import plot_multi_accel_updated as plot_multi


FIXED_PERSON_REP_COUNTS = {
    "Abhinav": 15,
    "Salvador": 15,
}

TRUTH_PATTERNS_SETS_X_REPS = [
    re.compile(r"(?i)(?<!\d)(\d+)x(\d+)(?!\d)"),
]

TRUTH_PATTERNS_REPS_ONLY = [
    re.compile(r"(?i)(?<!\d)(\d+)x(?:_|-|$)"),
]

ACTIVITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "pushups": {
        "candidate_signals": ["mag", "xy", "xz", "x"],
        "lowpass_hz": 2.176,
        "prominence_factor": 0.18,
        "min_dur_factor": 0.763,
        "min_period_s": 0.407,
        "max_period_s": 4.161,
        "prefer_devices": ["samsung_phone", "garmin_watch"],
    },
    "situps": {
        "candidate_signals": ["xy", "xz", "mag", "y"],
        "lowpass_hz": 4.6,
        "prominence_factor": 0.423,
        "min_dur_factor": 0.422,
        "min_period_s": 1.761,
        "max_period_s": 7.523,
        "prefer_devices": ["garmin_watch", "samsung_phone"],
    },
    "squats": {
        "candidate_signals": ["yz", "mag", "y", "xz"],
        "lowpass_hz": 2.661,
        "prominence_factor": 0.216,
        "min_dur_factor": 0.742,
        "min_period_s": 1.611,
        "max_period_s": 9.695,
        "prefer_devices": ["samsung_phone", "garmin_watch", "bose_headphones"],
    },
    "bench": {
        "candidate_signals": ["xy", "xz", "yz", "mag"],
        "lowpass_hz": 4.729,
        "prominence_factor": 0.397,
        "min_dur_factor": 0.79,
        "min_period_s": 0.896,
        "max_period_s": 8.858,
        "prefer_devices": ["garmin_watch", "samsung_phone"],
    },
    "lunges": {
        "candidate_signals": ["x", "yz", "mag", "xy"],
        "lowpass_hz": 4.091,
        "prominence_factor": 0.17,
        "min_dur_factor": 0.652,
        "min_period_s": 0.699,
        "max_period_s": 9.143,
        "prefer_devices": ["garmin_watch", "samsung_phone", "bose_headphones"],
    },
}

DEFAULT_ACTIVITY_CONFIG: Dict[str, Any] = {
    "candidate_signals": ["mag", "z", "y", "x"],
    "lowpass_hz": 4.0,
    "prominence_factor": 0.35,
    "min_dur_factor": 0.55,
    "min_period_s": 0.4,
    "max_period_s": 10.0,
    "prefer_devices": [],
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _session_metadata(root: Path, session_dir: Path) -> Dict[str, str]:
    rel = session_dir.relative_to(root)
    parts = rel.parts
    return {
        "relative_path": str(rel),
        "exercise": parts[0] if len(parts) >= 1 else "",
        "person": parts[1] if len(parts) >= 2 else "",
        "session": parts[-1] if parts else session_dir.name,
    }


def infer_expected_reps(meta: Dict[str, Any]) -> Optional[int]:
    person = str(meta.get("person", "")).strip()
    if person in FIXED_PERSON_REP_COUNTS:
        return int(FIXED_PERSON_REP_COUNTS[person])

    session_name = str(meta.get("session", "")).strip()
    for pattern in TRUTH_PATTERNS_SETS_X_REPS:
        match = pattern.search(session_name)
        if not match:
            continue
        sets = int(match.group(1))
        reps = int(match.group(2))
        if 1 <= sets <= 10 and 1 <= reps <= 40:
            return int(sets * reps)

    for pattern in TRUTH_PATTERNS_REPS_ONLY:
        match = pattern.search(session_name)
        if not match:
            continue
        reps = int(match.group(1))
        if 1 <= reps <= 40:
            return reps

    return None


def _canonical_exercise_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if key.endswith("es") and key[:-2] in {"lung", "bench"}:
        return key
    aliases = {
        "pushup": "pushups",
        "pushups": "pushups",
        "situp": "situps",
        "situps": "situps",
        "squat": "squats",
        "squats": "squats",
        "bench": "bench",
        "benchpress": "bench",
        "lunges": "lunges",
        "lunge": "lunges",
    }
    return aliases.get(key, key)


def _activity_config(exercise_name: str) -> Dict[str, Any]:
    return ACTIVITY_CONFIGS.get(_canonical_exercise_name(exercise_name), DEFAULT_ACTIVITY_CONFIG)


def _device_preference_bonus(label: str, prefer_devices: Sequence[str]) -> float:
    lname = str(label or "").lower()
    for rank, keyword in enumerate(prefer_devices):
        if keyword.lower() in lname:
            return float(len(prefer_devices) - rank)
    return 0.0


def _lowpass_signal(signal: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 5 or not np.isfinite(arr).any():
        return arr.copy()
    nyq = fs / 2.0
    cutoff = min(float(cutoff_hz), nyq * 0.90)
    if cutoff <= 0 or nyq <= 0:
        return arr.copy()
    b, a = butter(2, cutoff / nyq, btype="low")
    return filtfilt(b, a, arr)


def _candidate_signal_map(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> Dict[str, np.ndarray]:
    xx = np.asarray(xs, dtype=float)
    yy = np.asarray(ys, dtype=float)
    zz = np.asarray(zs, dtype=float)
    return {
        "x": xx,
        "y": yy,
        "z": zz,
        "mag": np.sqrt(xx**2 + yy**2 + zz**2),
        "xy": np.sqrt(xx**2 + yy**2),
        "xz": np.sqrt(xx**2 + zz**2),
        "yz": np.sqrt(yy**2 + zz**2),
    }


def _detect_events_on_signal(
    times: np.ndarray,
    signal: np.ndarray,
    *,
    lowpass_hz: float,
    prominence_factor: float,
    min_separation_s: Optional[float] = None,
    min_period_s: float = 0.3,
    max_period_s: float = 10.0,
) -> Dict[str, Any]:
    tt = np.asarray(times, dtype=float)
    xx = np.asarray(signal, dtype=float)
    if len(tt) < 5 or len(xx) < 5:
        return {
            "peak_idx": np.asarray([], dtype=int),
            "valley_idx": np.asarray([], dtype=int),
            "peak_times": np.asarray([], dtype=float),
            "valley_times": np.asarray([], dtype=float),
            "signal_lp": xx.copy(),
            "signal_bp": xx.copy(),
            "bp_low_hz": 0.0,
            "bp_high_hz": 0.0,
            "min_sep_used": 0.0,
        }

    dt = float(np.median(np.diff(tt)))
    fs = 1.0 / dt if dt > 0 else 100.0
    signal_lp = _lowpass_signal(xx, fs, lowpass_hz)
    centred = signal_lp - float(np.mean(signal_lp))

    if min_separation_s is None:
        period_s = plot_multi.estimate_rep_period_acf(
            centred,
            fs,
            min_period_s=min_period_s,
            max_period_s=max_period_s,
        )
        min_sep_used = period_s / 2.0
    else:
        period_s = min_separation_s * 2.0
        min_sep_used = min_separation_s

    signal_bp, bp_low_hz, bp_high_hz = plot_multi.bandpass_filter(centred, fs, period_s)
    dist_samples = max(1, int(min_sep_used * fs))
    iqr = float(np.percentile(signal_bp, 75) - np.percentile(signal_bp, 25))
    prominence = max(iqr * float(prominence_factor), 1e-6)

    peak_idx, _ = find_peaks(signal_bp, distance=dist_samples, prominence=prominence)
    valley_idx, _ = find_peaks(-signal_bp, distance=dist_samples, prominence=prominence)

    return {
        "peak_idx": peak_idx,
        "valley_idx": valley_idx,
        "peak_times": tt[peak_idx],
        "valley_times": tt[valley_idx],
        "signal_lp": signal_lp,
        "signal_bp": signal_bp,
        "bp_low_hz": float(bp_low_hz),
        "bp_high_hz": float(bp_high_hz),
        "min_sep_used": float(min_sep_used),
    }


def _segment_regularity(segment_windows: Sequence[Tuple[float, float]]) -> float:
    durations = np.asarray([max(float(end_s - start_s), 0.0) for start_s, end_s in segment_windows], dtype=float)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    if len(durations) < 2:
        return 1.0
    return float(np.std(durations) / max(np.mean(durations), 1e-6))


def _rolling_mean(arr: np.ndarray, n: int) -> np.ndarray:
    xx = np.asarray(arr, dtype=float)
    if len(xx) == 0:
        return xx.copy()
    n = max(1, min(int(n), len(xx)))
    if n <= 1:
        return xx.copy()
    kernel = np.ones(n, dtype=float) / float(n)
    pad_left = n // 2
    pad_right = n - 1 - pad_left
    return np.convolve(np.pad(xx, (pad_left, pad_right), mode="edge"), kernel, mode="valid")


def _active_movement_window(
    times: np.ndarray,
    signal_bp: np.ndarray,
    *,
    min_duration_s: float = 6.0,
    bridge_gap_s: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """
    Find the dominant sustained high-energy movement block in a filtered trace.

    Some captures contain setup motion before the real set. Pure regularity
    scoring can prefer that short setup block. This picks the contiguous region
    with the strongest sustained bandpassed energy so downstream rep windows can
    be evaluated inside the actual exercise block.
    """
    tt = np.asarray(times, dtype=float)
    xx = np.asarray(signal_bp, dtype=float)
    if len(tt) < 5 or len(xx) < 5 or len(tt) != len(xx):
        return None

    finite = np.isfinite(tt) & np.isfinite(xx)
    if not np.any(finite):
        return None
    tt = tt[finite]
    xx = xx[finite]
    if len(tt) < 5:
        return None

    dt = float(np.median(np.diff(tt)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    fs = 1.0 / dt
    envelope = _rolling_mean(np.abs(xx), max(3, int(round(1.0 * fs))))
    if len(envelope) != len(tt) or not np.isfinite(envelope).any():
        return None

    hi = float(np.percentile(envelope, 92))
    med = float(np.median(envelope))
    if hi <= med:
        return None
    threshold = med + 0.35 * (hi - med)
    active = envelope >= threshold
    active_idx = np.flatnonzero(active)
    if len(active_idx) == 0:
        return None

    raw_runs: List[Tuple[int, int]] = []
    start = int(active_idx[0])
    prev = int(active_idx[0])
    max_gap_samples = max(1, int(round(bridge_gap_s * fs)))
    for idx in active_idx[1:]:
        idx = int(idx)
        if idx - prev > max_gap_samples:
            raw_runs.append((start, prev))
            start = idx
        prev = idx
    raw_runs.append((start, prev))

    candidates: List[Dict[str, Any]] = []
    for start_i, end_i in raw_runs:
        start_s = float(tt[start_i])
        end_s = float(tt[end_i])
        duration_s = max(0.0, end_s - start_s)
        if duration_s < min_duration_s:
            continue
        seg = envelope[start_i : end_i + 1]
        mean_energy = float(np.mean(seg)) if len(seg) else 0.0
        p95_energy = float(np.percentile(seg, 95)) if len(seg) else 0.0
        candidates.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": duration_s,
                "mean_energy": mean_energy,
                "p95_energy": p95_energy,
                "score": float(mean_energy * np.sqrt(max(duration_s, 1e-6))),
                "threshold": threshold,
            }
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item["score"]))


def _filter_segments_to_active_window(
    segments: Sequence[Tuple[int, int]],
    segment_windows: Sequence[Tuple[float, float]],
    active_window: Optional[Dict[str, Any]],
    *,
    min_keep: int = 3,
) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]], Dict[str, Any]]:
    if active_window is None:
        return list(segments), list(segment_windows), {"applied": False, "reason": "no_active_window"}

    start_s = float(active_window["start_s"])
    end_s = float(active_window["end_s"])
    kept_segments: List[Tuple[int, int]] = []
    kept_windows: List[Tuple[float, float]] = []
    for segment, window in zip(segments, segment_windows):
        w_start, w_end = float(window[0]), float(window[1])
        center = 0.5 * (w_start + w_end)
        if start_s <= center <= end_s:
            kept_segments.append((int(segment[0]), int(segment[1])))
            kept_windows.append((w_start, w_end))

    if len(kept_windows) < min_keep:
        return (
            list(segments),
            list(segment_windows),
            {
                "applied": False,
                "reason": "too_few_segments_in_active_window",
                "active_window": active_window,
                "kept_count": len(kept_windows),
                "original_count": len(segment_windows),
            },
        )

    return (
        kept_segments,
        kept_windows,
        {
            "applied": True,
            "reason": "filtered_to_active_window",
            "active_window": active_window,
            "kept_count": len(kept_windows),
            "original_count": len(segment_windows),
        },
    )


def _resample_segment_for_similarity(segment: np.ndarray, n_target: int = 96) -> np.ndarray:
    xx = np.asarray(segment, dtype=float)
    if len(xx) == 0:
        return np.zeros(n_target, dtype=float)
    if len(xx) == 1:
        return np.full(n_target, float(xx[0]), dtype=float)

    old = np.linspace(0.0, 1.0, len(xx), dtype=float)
    new = np.linspace(0.0, 1.0, int(n_target), dtype=float)
    yy = np.interp(new, old, xx)
    std = float(np.std(yy))
    if std <= 1e-9:
        return yy - float(np.mean(yy))
    return (yy - float(np.mean(yy))) / std


def _mean_rep_correlation(
    segment_windows: Sequence[Tuple[float, float]],
    times: np.ndarray,
    signal: np.ndarray,
    target_index: int,
    reference_indices: Sequence[int],
) -> float:
    tt = np.asarray(times, dtype=float)
    xx = np.asarray(signal, dtype=float)
    if not (0 <= target_index < len(segment_windows)) or len(tt) == 0 or len(xx) == 0:
        return float("nan")

    target_start, target_end = segment_windows[target_index]
    target = plot_multi.extract_segment_by_time(tt, xx, target_start, target_end)
    target_r = _resample_segment_for_similarity(target)
    if float(np.std(target_r)) <= 1e-9:
        return float("nan")

    vals: List[float] = []
    for ref_i in reference_indices:
        if not (0 <= int(ref_i) < len(segment_windows)):
            continue
        ref_start, ref_end = segment_windows[int(ref_i)]
        ref = plot_multi.extract_segment_by_time(tt, xx, ref_start, ref_end)
        ref_r = _resample_segment_for_similarity(ref)
        if float(np.std(ref_r)) <= 1e-9:
            continue
        corr = float(np.corrcoef(target_r, ref_r)[0, 1])
        if np.isfinite(corr):
            vals.append(corr)

    return float(np.mean(vals)) if vals else float("nan")


def _drop_bad_terminal_rep(
    segment_windows: Sequence[Tuple[float, float]],
    times: np.ndarray,
    signal: np.ndarray,
    *,
    min_reps: int = 5,
    min_abs_gap: float = 0.10,
    max_ratio: float = 0.75,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Drop the final rep if it correlates much worse with prior reps than the
    second-final rep does.

    This is intentionally limited to the terminal rep. It catches common
    trailing partial-motion artifacts without changing interior boundaries.
    """
    windows = [(float(s), float(e)) for s, e in segment_windows if float(e) > float(s)]
    info: Dict[str, Any] = {
        "applied": False,
        "dropped": False,
        "last_mean_corr": None,
        "second_last_mean_corr": None,
        "last_duration_s": None,
        "min_previous_duration_s": None,
        "reason": "",
    }
    n = len(windows)
    if n < min_reps:
        info["reason"] = "too_few_reps"
        return windows, info

    last_refs = list(range(0, n - 1))
    second_last_refs = list(range(0, n - 2))
    if not second_last_refs:
        info["reason"] = "too_few_reference_reps"
        return windows, info

    durations = np.asarray([max(float(end_s - start_s), 0.0) for start_s, end_s in windows], dtype=float)
    last_duration = float(durations[-1])
    previous_durations = durations[:-1]
    min_previous_duration = float(np.min(previous_durations)) if len(previous_durations) else float("nan")
    info["last_duration_s"] = last_duration
    info["min_previous_duration_s"] = min_previous_duration
    if np.isfinite(min_previous_duration) and last_duration < min_previous_duration:
        info["applied"] = True
        info["dropped"] = True
        info["reason"] = "last_rep_shorter_than_all_previous_reps"
        return windows[:-1], info

    last_score = _mean_rep_correlation(windows, times, signal, n - 1, last_refs)
    second_last_score = _mean_rep_correlation(windows, times, signal, n - 2, second_last_refs)
    info.update(
        {
            "applied": True,
            "last_mean_corr": float(last_score) if np.isfinite(last_score) else None,
            "second_last_mean_corr": float(second_last_score) if np.isfinite(second_last_score) else None,
        }
    )

    if not (np.isfinite(last_score) and np.isfinite(second_last_score)):
        info["reason"] = "non_finite_correlation"
        return windows, info

    abs_gap = float(second_last_score - last_score)
    ratio = float(last_score / max(second_last_score, 1e-6)) if second_last_score > 0 else float("inf")
    info["absolute_gap"] = abs_gap
    info["ratio"] = ratio

    if abs_gap >= min_abs_gap and ratio <= max_ratio:
        info["dropped"] = True
        info["reason"] = "last_rep_significantly_worse_than_second_last"
        return windows[:-1], info

    info["reason"] = "last_rep_not_significantly_worse"
    return windows, info


def _choose_activity_candidate(
    *,
    exercise_name: str,
    label: str,
    times: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    expected_reps: Optional[int],
    template_reps: int,
) -> Dict[str, Any]:
    # expected_reps is retained for API compatibility, but is intentionally not
    # used for selecting the signal/event. Boundary detection should be driven by
    # signal quality, not by a filename-derived target count.
    _ = expected_reps
    config = _activity_config(exercise_name)
    exercise_key = _canonical_exercise_name(exercise_name)
    candidates = _candidate_signal_map(xs, ys, zs)
    best: Optional[Dict[str, Any]] = None
    prefer_bonus = _device_preference_bonus(label, config.get("prefer_devices", []))

    for signal_name in config["candidate_signals"]:
        signal = candidates.get(signal_name)
        if signal is None:
            continue
        detected = _detect_events_on_signal(
            np.asarray(times, dtype=float),
            np.asarray(signal, dtype=float),
            lowpass_hz=float(config["lowpass_hz"]),
            prominence_factor=float(config["prominence_factor"]),
            min_separation_s=None,
            min_period_s=float(config.get("min_period_s", 0.3)),
            max_period_s=float(config.get("max_period_s", 10.0)),
        )
        for event_name in ("valley", "peak"):
            idx = detected[f"{event_name}_idx"]
            segments = plot_multi.segment_reps(
                idx,
                n_samples=len(signal),
                min_dur_factor=float(config["min_dur_factor"]),
                n_template=template_reps,
            )
            segment_windows = plot_multi.segments_to_time_windows(segments, times)
            active_window = None
            segments, segment_windows, active_filter = _filter_segments_to_active_window(
                segments,
                segment_windows,
                active_window,
            )
            seg_count = len(segment_windows)
            regularity = _segment_regularity(segment_windows)
            signal_iqr = float(np.percentile(detected["signal_bp"], 75) - np.percentile(detected["signal_bp"], 25))
            if exercise_key == "squats" and prefer_bonus > 0:
                score = (
                    prefer_bonus,
                    float(seg_count),
                    -regularity,
                    signal_iqr,
                )
            else:
                score = (
                    -regularity,
                    prefer_bonus,
                    signal_iqr,
                    float(seg_count),
                )
            candidate = {
                "selected_signal": signal_name,
                "selected_event": event_name,
                "segments": segments,
                "segment_windows": segment_windows,
                "segment_count": seg_count,
                "regularity": regularity,
                "peak_times": detected["peak_times"],
                "valley_times": detected["valley_times"],
                "peak_idx": detected["peak_idx"],
                "valley_idx": detected["valley_idx"],
                "signal_lp": detected["signal_lp"],
                "signal_bp": detected["signal_bp"],
                "bp_low_hz": detected["bp_low_hz"],
                "bp_high_hz": detected["bp_high_hz"],
                "min_sep_used": detected["min_sep_used"],
                "active_filter": active_filter,
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        fallback_peak_times, fallback_valley_times, fallback_valley_idx, mag_f, mag_bp, bp_low_hz, bp_high_hz, min_sep_used = plot_multi.detect_peaks_valleys(
            np.asarray(times, dtype=float),
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
            np.asarray(zs, dtype=float),
            lowpass_hz=float(config["lowpass_hz"]),
            min_separation_s=None,
            prominence_factor=float(config["prominence_factor"]),
        )
        segments = plot_multi.segment_reps(
            fallback_valley_idx,
            n_samples=len(mag_f),
            min_dur_factor=float(config["min_dur_factor"]),
            n_template=template_reps,
        )
        segment_windows = plot_multi.segments_to_time_windows(segments, times)
        active_window = None
        segments, segment_windows, active_filter = _filter_segments_to_active_window(
            segments,
            segment_windows,
            active_window,
        )
        best = {
            "selected_signal": "mag",
            "selected_event": "valley",
            "segments": segments,
            "segment_windows": segment_windows,
            "segment_count": len(segments),
            "regularity": _segment_regularity(segment_windows),
            "peak_times": fallback_peak_times,
            "valley_times": fallback_valley_times,
            "peak_idx": np.asarray([], dtype=int),
            "valley_idx": fallback_valley_idx,
            "signal_lp": mag_f,
            "signal_bp": mag_bp,
            "bp_low_hz": float(bp_low_hz),
            "bp_high_hz": float(bp_high_hz),
            "min_sep_used": float(min_sep_used),
            "active_filter": active_filter,
            "score": (0.0, 0.0, 0.0, 0.0),
        }
    return best


def _is_finalrep_session(session_dir: Path) -> bool:
    return any((session_dir / name).exists() for name in finalrep.SENSOR_FILES.values())


def _is_plot_multi_session(session_dir: Path) -> bool:
    try:
        return len(plot_multi.discover_device_csvs(session_dir)) > 0
    except Exception:
        return False


def _is_named_multi_device_session(session_dir: Path) -> bool:
    device_markers = (
        "bose_headphones",
        "garmin_watch",
        "samsung_phone",
    )
    csv_names = [p.name.lower() for p in session_dir.glob("*.csv")]
    return any(any(marker in name for marker in device_markers) for name in csv_names)


def _preferred_analyzer(session_dir: Path) -> str:
    if _is_named_multi_device_session(session_dir) and _is_plot_multi_session(session_dir):
        return "plot_multi_accel_updated"
    if _is_finalrep_session(session_dir):
        return "FINALREP"
    if _is_plot_multi_session(session_dir):
        return "plot_multi_accel_updated"
    raise ValueError(f"No supported analyzer found for {session_dir}")


def _iter_session_dirs(root: Path) -> Iterable[Path]:
    candidates = [root] + sorted(p for p in root.rglob("*") if p.is_dir())
    for path in candidates:
        if not any(path.glob("*.csv")):
            continue
        if _is_finalrep_session(path) or _is_plot_multi_session(path):
            yield path


def _peak_windows(events: np.ndarray, period_s: float) -> List[Tuple[float, float, float]]:
    ev = np.asarray(events, dtype=float)
    ev = ev[np.isfinite(ev)]
    ev = np.unique(np.sort(ev))
    if len(ev) == 0:
        return []
    if len(ev) == 1:
        c = float(ev[0])
        return [(c - 0.5 * period_s, c + 0.5 * period_s, c)]

    out: List[Tuple[float, float, float]] = []
    for i, center in enumerate(ev):
        if i == 0:
            left = float(center - 0.5 * (ev[1] - ev[0]))
        else:
            left = float(0.5 * (ev[i - 1] + ev[i]))

        if i == (len(ev) - 1):
            right = float(center + 0.5 * (ev[-1] - ev[-2]))
        else:
            right = float(0.5 * (ev[i] + ev[i + 1]))
        out.append((left, right, float(center)))
    return out


def _frac_between_peaks(peaks_sorted: np.ndarray, t0: float) -> Optional[float]:
    if len(peaks_sorted) < 2:
        return None
    idx = int(np.searchsorted(peaks_sorted, t0) - 1)
    j = idx + 1
    if 0 <= idx < len(peaks_sorted) and 0 <= j < len(peaks_sorted):
        left = float(peaks_sorted[idx])
        right = float(peaks_sorted[j])
        if right > left:
            return float((t0 - left) / (right - left))
    return None


def _build_finalrep_windows_for_set(
    peaks: np.ndarray,
    valleys: np.ndarray,
    period_s: float,
    st: Dict[str, Any],
) -> List[Dict[str, Any]]:
    start_s = float(st["start_s"])
    end_s = float(st["end_s"])
    target_reps = int(st.get("rep_count", 0))

    pad_s = max(0.30, min(0.60 * period_s, 2.40))
    win_lo = start_s - pad_s
    win_hi = end_s + pad_s

    peak_wins = [
        (l, r, c)
        for (l, r, c) in _peak_windows(peaks, period_s)
        if max(l, win_lo) <= min(r, win_hi)
    ]
    peak_with_valley = [
        (l, r, c)
        for (l, r, c) in peak_wins
        if bool(np.any((valleys >= l) & (valleys <= r)))
    ]

    valley_support = (float(len(peak_with_valley)) / float(len(peak_wins))) if peak_wins else 0.0
    if peak_wins and valley_support >= 0.60:
        candidate_centers = [float(c) for (_, _, c) in peak_wins]
    else:
        src = peak_with_valley if peak_with_valley else peak_wins
        candidate_centers = [float(c) for (_, _, c) in src]

    duration_est = max(1, int(round((end_s - start_s) / period_s)) + 1)
    if (target_reps > len(candidate_centers)) and duration_est == (len(candidate_centers) + 1) and len(peaks) >= 3:
        start_frac = _frac_between_peaks(peaks, start_s)
        end_frac = _frac_between_peaks(peaks, end_s)
        start_cut = bool(start_frac is not None and start_frac >= 0.68)
        end_cut = bool(end_frac is not None and end_frac <= 0.32)

        edge_band = 0.42 * period_s
        edge_valley = bool(
            np.any((valleys >= (start_s - edge_band)) & (valleys <= (start_s + edge_band)))
            or np.any((valleys >= (end_s - edge_band)) & (valleys <= (end_s + edge_band)))
        )
        if edge_valley and (start_cut or end_cut):
            if start_cut and not end_cut:
                candidate_centers.append(float(start_s - 0.25 * period_s))
            elif end_cut and not start_cut:
                candidate_centers.append(float(end_s + 0.25 * period_s))
            else:
                start_excess = float(start_frac - 0.5) if start_frac is not None else 0.0
                end_excess = float(0.5 - end_frac) if end_frac is not None else 0.0
                if start_excess >= end_excess:
                    candidate_centers.append(float(start_s - 0.25 * period_s))
                else:
                    candidate_centers.append(float(end_s + 0.25 * period_s))

    centers = finalrep._regularize_rep_centers_by_distance(
        centers=candidate_centers,
        start_s=start_s,
        end_s=end_s,
        target_reps=target_reps,
        period_s=period_s,
    )
    if not centers:
        return []

    valleys_set = valleys[(valleys >= (start_s - 0.80 * period_s)) & (valleys <= (end_s + 0.80 * period_s))]
    if len(valleys_set):
        counted = finalrep._valley_anchored_windows_from_centers(centers, valleys_set, period_s)
    else:
        counted = finalrep._midpoint_windows_from_centers(centers, period_s)

    raw = np.asarray([float(c) for c in candidate_centers if np.isfinite(c)], dtype=float)
    raw = np.unique(np.sort(raw))
    step_s = finalrep._estimate_rep_spacing(centers, start_s, end_s, max(target_reps, len(centers)), period_s)
    match_tol = max(0.20 * step_s, 0.22 * period_s)

    out: List[Dict[str, Any]] = []
    for (left, right, center) in counted:
        near_raw = bool(len(raw) and np.min(np.abs(raw - float(center))) <= match_tol)
        out.append(
            {
                "left": float(left),
                "right": float(right),
                "center": float(center),
                "synthetic": bool(not near_raw),
                "set_start": float(start_s),
                "set_end": float(end_s),
            }
        )
    return out


def _extract_finalrep_rep_records(session_dir: Path, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected = finalrep._get_selected_channels(session_dir)
    sets = result.get("sets", [])
    if not selected or not sets:
        return []

    best = selected[0]
    peaks = np.asarray(best.peaks_t, dtype=float)
    peaks = peaks[np.isfinite(peaks)]
    peaks = np.unique(np.sort(peaks))

    valleys = np.asarray(best.valleys_t, dtype=float)
    valleys = valleys[np.isfinite(valleys)]
    valleys = np.unique(np.sort(valleys))

    period_s = max(float(best.period_s), 1e-6)

    rep_windows_raw: List[Dict[str, Any]] = []
    set_lookup: Dict[Tuple[float, float], int] = {}
    for set_index, st in enumerate(sets, start=1):
        key = (float(st["start_s"]), float(st["end_s"]))
        set_lookup[key] = set_index
        rep_windows_raw.extend(_build_finalrep_windows_for_set(peaks, valleys, period_s, st))

    rep_windows = finalrep._standardize_windows_by_cadence(
        rep_windows=rep_windows_raw,
        consensus_sets=sets,
        period_s=period_s,
        valley_times=valleys,
    )
    if not rep_windows:
        return []

    start_bounds = np.asarray([float(w["left"]) for w in rep_windows], dtype=float)
    ends = np.asarray([float(w["right"]) for w in rep_windows], dtype=float)
    centers = np.asarray([float(w["center"]) for w in rep_windows], dtype=float)
    starts = finalrep._estimate_peak_aligned_starts(rep_windows, peaks)
    if len(starts) != len(start_bounds):
        starts = finalrep._estimate_rep_onset_starts(rep_windows)
    if len(starts) != len(start_bounds):
        starts = start_bounds.copy()
    starts = finalrep._apply_set_start_anchors_to_starts(starts, centers, rep_windows, sets)

    reps: List[Dict[str, Any]] = []
    for rep_index, (w, start_s, end_s, center_s, boundary_start_s) in enumerate(
        zip(rep_windows, starts, ends, centers, start_bounds),
        start=1,
    ):
        set_key = (float(w.get("set_start", np.nan)), float(w.get("set_end", np.nan)))
        set_index = set_lookup.get(set_key)
        reps.append(
            {
                "rep_index": rep_index,
                "start_s": float(start_s),
                "end_s": float(end_s),
                "center_s": float(center_s),
                "duration_s": float(max(end_s - start_s, 0.0)),
                "boundary_start_s": float(boundary_start_s),
                "set_index": set_index,
                "synthetic": bool(w.get("synthetic", False)),
            }
        )
    return reps


def _fallback_valley_rep_records(valleys_t: Sequence[float]) -> List[Dict[str, Any]]:
    vv = np.asarray([float(v) for v in valleys_t if _safe_float(v) is not None], dtype=float)
    vv = np.unique(np.sort(vv))
    reps: List[Dict[str, Any]] = []
    for rep_index, (left, right) in enumerate(zip(vv[:-1], vv[1:]), start=1):
        if right <= left:
            continue
        reps.append(
            {
                "rep_index": rep_index,
                "start_s": float(left),
                "end_s": float(right),
                "center_s": float(0.5 * (left + right)),
                "duration_s": float(right - left),
                "boundary_start_s": float(left),
                "set_index": None,
                "synthetic": False,
            }
        )
    return reps


def _apply_terminal_prune_to_rep_records(
    reps: Sequence[Dict[str, Any]],
    times: np.ndarray,
    signal: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    windows = [
        (float(rep["start_s"]), float(rep["end_s"]))
        for rep in reps
        if _safe_float(rep.get("start_s")) is not None and _safe_float(rep.get("end_s")) is not None
    ]
    pruned_windows, info = _drop_bad_terminal_rep(windows, times, signal)
    if len(pruned_windows) == len(windows):
        return [dict(rep) for rep in reps], info

    keep_count = len(pruned_windows)
    pruned_reps: List[Dict[str, Any]] = []
    for i, rep in enumerate(list(reps)[:keep_count], start=1):
        item = dict(rep)
        item["rep_index"] = i
        pruned_reps.append(item)
    return pruned_reps, info


def analyze_with_finalrep(session_dir: Path) -> Dict[str, Any]:
    result = finalrep.analyze_session(session_dir)
    reps = _extract_finalrep_rep_records(session_dir, result)
    selected = finalrep._get_selected_channels(session_dir)
    if not reps:
        if selected:
            reps = _fallback_valley_rep_records(selected[0].valleys_t)
    terminal_prune: Dict[str, Any] = {
        "applied": False,
        "dropped": False,
        "last_mean_corr": None,
        "second_last_mean_corr": None,
        "reason": "no_selected_channel_or_reps",
    }
    if reps and selected:
        best = selected[0]
        reps, terminal_prune = _apply_terminal_prune_to_rep_records(
            reps,
            np.asarray(best.trace_t, dtype=float),
            np.asarray(best.trace_filt, dtype=float),
        )

    estimated_total_reps = len(reps) if reps else int(result.get("estimated_total_reps", 0))
    return {
        "analyzer": "FINALREP",
        "estimated_total_reps": int(estimated_total_reps),
        "model_estimated_total_reps": int(result.get("estimated_total_reps", 0)),
        "estimated_proper_sets": int(result.get("estimated_proper_sets", 0)),
        "exercise_label": str(result.get("exercise", "")),
        "sets": result.get("sets", []),
        "top_channels": result.get("top_channels", []),
        "reps": reps,
        "terminal_rep_prune": terminal_prune,
        "source_files": sorted(p.name for p in session_dir.glob("*.csv")),
    }


def _choose_primary_device(
    devices: Sequence[Dict[str, Any]],
    template_reps: int,
    expected_reps: Optional[int] = None,
    exercise_name: str = "",
) -> int:
    # expected_reps is retained for API compatibility, but is intentionally not
    # used for primary-device selection.
    _ = expected_reps
    exercise_key = _canonical_exercise_name(exercise_name)
    viable_idxs = [i for i, d in enumerate(devices) if len(d.get("segments") or []) >= (template_reps + 1)]
    candidate_idxs = viable_idxs if viable_idxs else list(range(len(devices)))

    def _score(i: int) -> Tuple[float, float, float, float, float]:
        seg_count = len(devices[i].get("segments") or [])
        times = np.asarray(devices[i].get("times", []), dtype=float)
        duration = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
        fs = float(len(times) / duration) if duration > 0 else 0.0
        segment_windows = plot_multi.segments_to_time_windows(devices[i].get("segments") or [], times)
        rep_durations = np.asarray([max(float(end_s - start_s), 0.0) for start_s, end_s in segment_windows], dtype=float)
        rep_durations = rep_durations[np.isfinite(rep_durations) & (rep_durations > 0)]
        if len(rep_durations) >= 2:
            cv = float(np.std(rep_durations) / max(np.mean(rep_durations), 1e-6))
        else:
            cv = 1.0
        regularity_score = -cv
        duration_score = float(np.median(rep_durations)) if len(rep_durations) else 0.0
        preference_score = float(devices[i].get("device_preference", 0.0)) if exercise_key == "squats" else 0.0
        return (preference_score, regularity_score, duration_score, float(seg_count), fs)

    return max(candidate_idxs, key=_score)


def analyze_with_plot_multi(
    session_dir: Path,
    template_reps: int = 5,
    expected_reps: Optional[int] = None,
    exercise_name: str = "",
) -> Dict[str, Any]:
    input_files = plot_multi.resolve_input_paths([str(session_dir)])
    labels = [Path(f).stem for f in input_files]

    all_data = []
    for filepath, label in zip(input_files, labels):
        ts, xs, ys, zs = plot_multi.load_device_csv(filepath)
        all_data.append({"label": label, "ts": ts, "xs": xs, "ys": ys, "zs": zs, "source_path": str(filepath)})

    t_start = max(d["ts"][0] for d in all_data) + 2
    t_end = min(d["ts"][-1] for d in all_data) - 2
    if t_start >= t_end:
        raise ValueError(f"No overlapping time window found for {session_dir}")

    devices: List[Dict[str, Any]] = []
    for d in all_data:
        ts, xs, ys, zs = plot_multi.trim_to_window(d["ts"], d["xs"], d["ys"], d["zs"], t_start, t_end)
        times = ts - ts[0]
        activity_choice = _choose_activity_candidate(
            exercise_name=exercise_name,
            label=d["label"],
            times=times,
            xs=xs,
            ys=ys,
            zs=zs,
            expected_reps=expected_reps,
            template_reps=template_reps,
        )
        devices.append(
            {
                "label": d["label"],
                "source_path": d["source_path"],
                "times": times,
                "xs": xs,
                "ys": ys,
                "zs": zs,
                "peak_times": activity_choice["peak_times"],
                "valley_times": activity_choice["valley_times"],
                "valley_idx": activity_choice["valley_idx"],
                "mag_f": activity_choice["signal_lp"],
                "mag_bp": activity_choice["signal_bp"],
                "bp_low_hz": activity_choice["bp_low_hz"],
                "bp_high_hz": activity_choice["bp_high_hz"],
                "min_sep_used": activity_choice["min_sep_used"],
                "segments": activity_choice["segments"],
                "selected_signal": str(activity_choice["selected_signal"]),
                "selected_event": str(activity_choice["selected_event"]),
                "regularity": float(activity_choice["regularity"]),
                "device_preference": float(_device_preference_bonus(d["label"], _activity_config(exercise_name).get("prefer_devices", []))),
                "active_filter": activity_choice.get("active_filter", {}),
            }
        )

    primary_idx = _choose_primary_device(
        devices,
        template_reps=template_reps,
        expected_reps=expected_reps,
        exercise_name=exercise_name,
    )
    primary = devices[primary_idx]
    segment_windows = plot_multi.segments_to_time_windows(primary["segments"], primary["times"])
    segment_windows, terminal_prune = _drop_bad_terminal_rep(
        segment_windows,
        np.asarray(primary["times"], dtype=float),
        np.asarray(primary["mag_bp"], dtype=float),
    )

    reps = [
        {
            "rep_index": i,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "center_s": float(0.5 * (start_s + end_s)),
            "duration_s": float(end_s - start_s),
            "boundary_start_s": float(start_s),
            "set_index": None,
            "synthetic": False,
        }
        for i, (start_s, end_s) in enumerate(segment_windows, start=1)
        if end_s > start_s
    ]

    return {
        "analyzer": "plot_multi_accel_updated",
        "estimated_total_reps": int(len(reps)),
        "model_estimated_total_reps": int(len(reps)),
        "estimated_proper_sets": 0,
        "exercise_label": "",
        "sets": [],
        "top_channels": [],
        "reps": reps,
        "primary_device": str(primary["label"]),
        "primary_device_index": int(primary_idx),
        "activity_strategy": f"{_canonical_exercise_name(exercise_name)}:{primary.get('selected_signal', 'mag')}:{primary.get('selected_event', 'valley')}",
        "active_filter": primary.get("active_filter", {}),
        "terminal_rep_prune": terminal_prune,
        "source_files": [Path(p).name for p in input_files],
    }


def _choose_best_analysis(
    analyses: Sequence[Dict[str, Any]],
    expected_reps: Optional[int],
    preferred_analyzer: str,
) -> Dict[str, Any]:
    if not analyses:
        raise ValueError("No analyses available to choose from")
    if expected_reps is None:
        for analysis in analyses:
            if analysis.get("analyzer") == preferred_analyzer:
                return analysis
        return analyses[0]

    def _score(analysis: Dict[str, Any]) -> Tuple[int, int]:
        estimated = int(analysis.get("estimated_total_reps", 0))
        return (abs(estimated - expected_reps), 0 if analysis.get("analyzer") == preferred_analyzer else 1)

    return min(analyses, key=_score)


def _summary_row(session: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "relative_path": session["relative_path"],
        "exercise": session["exercise"],
        "person": session["person"],
        "session": session["session"],
        "analyzer": session["analyzer"],
        "estimated_total_reps": session["estimated_total_reps"],
        "model_estimated_total_reps": session["model_estimated_total_reps"],
        "rep_boundary_count": len(session.get("reps", [])),
        "estimated_proper_sets": session.get("estimated_proper_sets", 0),
        "primary_device": session.get("primary_device", ""),
        "activity_strategy": session.get("activity_strategy", ""),
        "exercise_label": session.get("exercise_label", ""),
    }


def _rep_rows(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rep in session.get("reps", []):
        out.append(
            {
                "relative_path": session["relative_path"],
                "exercise": session["exercise"],
                "person": session["person"],
                "session": session["session"],
                "analyzer": session["analyzer"],
                "rep_index": rep["rep_index"],
                "start_s": rep["start_s"],
                "end_s": rep["end_s"],
                "center_s": rep["center_s"],
                "duration_s": rep["duration_s"],
                "boundary_start_s": rep.get("boundary_start_s"),
                "set_index": rep.get("set_index"),
                "synthetic": rep.get("synthetic", False),
            }
        )
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch rep count and boundary extraction across nested exercise folders.")
    parser.add_argument("root", type=str, help="Dataset root directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for JSON/CSV outputs")
    parser.add_argument("--plots", action="store_true", help="Generate diagnostic plots for each session (using FINALREP's plotting for FINALREP sessions)")
    parser.add_argument("--template-reps", type=int, default=5, help="Template rep count for plot_multi-based sessions")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "rep_analysis_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # Store analysis results for plotting later, if requested
    analysis_results_for_plotting: List[Tuple[Path, str, Optional[Dict[str, Any]]]] = []


    for session_dir in _iter_session_dirs(root):
        meta = _session_metadata(root, session_dir)
        expected_reps = infer_expected_reps(meta)
        print(f"Analyzing {meta['relative_path']} ...")
        try:
            analysis = analyze_with_plot_multi(
                session_dir,
                template_reps=args.template_reps,
                expected_reps=expected_reps,
                exercise_name=meta["exercise"],
            )

            record = {
                **meta,
                **analysis,
            }
            sessions.append(record)
            print(f"  -> {record['analyzer']}: reps={record['estimated_total_reps']}, boundaries={len(record.get('reps', []))}")

            if args.plots:
                session_plots_dir = out_dir / meta["relative_path"] / "analysis_plots"
                session_plots_dir.mkdir(parents=True, exist_ok=True)
                if record["analyzer"] == "FINALREP":
                    finalrep.generate_session_graphs(session_dir, out_dir=session_plots_dir)
                else:
                    print(f"    Note: Batch plotting for {record['analyzer']} is not yet implemented.")

        except Exception as exc:
            err = {
                **meta,
                "error": str(exc),
            }
            errors.append(err)
            print(f"  -> ERROR: {exc}")

    summary_rows = [_summary_row(s) for s in sessions]
    rep_rows: List[Dict[str, Any]] = []
    for session in sessions:
        rep_rows.extend(_rep_rows(session))

    report = {
        "root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_count": len(sessions),
        "error_count": len(errors),
        "sessions": sessions,
        "errors": errors,
    }

    json_path = out_dir / "rep_report.json"
    summary_csv = out_dir / "session_summary.csv"
    reps_csv = out_dir / "rep_boundaries.csv"

    json_path.write_text(json.dumps(report, indent=2, default=_json_default))
    _write_csv(summary_csv, summary_rows)
    _write_csv(reps_csv, rep_rows)

    print("")
    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote session summary CSV: {summary_csv}")
    print(f"Wrote rep boundary CSV: {reps_csv}")
    print(f"Sessions analyzed: {len(sessions)}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
