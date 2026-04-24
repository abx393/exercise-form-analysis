#!/usr/bin/env python3
"""
FINAL rep/set analysis for wearable IMU CSV sessions.

This script is designed for exported session folders containing files like:
  Accelerometer.csv, Gyroscope.csv, Gravity.csv, Magnetometer.csv, Orientation.csv

It combines the strongest parts of prior approaches:
  1) Scoring every sensor-axis channel for periodicity quality.
  2) Estimating cadence per channel (Welch PSD) and adapting peak constraints.
  3) Jerk-based active-window trimming to remove setup/rack dead space.
  4) Detecting valley events (rep anchors) with envelope gating + de-duplication.
  5) Cross-axis trust weighting to reduce noisy sensor-axis influence.
  6) Splitting reps into sets by cadence-scaled temporal gaps.
  7) Clustering set candidates across top channels for consensus set counts.
  8) Peak/valley hybrid rep refinement + boundary compensation + cadence regularization.
  9) Template matching support via cross-correlation, DTW, and subsequence search.

Typical usage:
  python improved_rep_set_analysis.py --root "Tuesday March 3rd"
  python improved_rep_set_analysis.py --session "Tuesday March 3rd/squat_8x_115-..."
  python improved_rep_set_analysis.py --drive-session "College/Senior/Bio Sensing/FinalProject/Tuesday March 3rd/squat_8x_115-..."
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks, hilbert, welch

# ----------------------------- Tunables ------------------------------------- #
TIME_ALIASES = [
    "seconds_elapsed",
    "time",
    "Time",
    "timestamp",
    "Timestamp",
    "absolute_timestamp",
    "elapsed",
    "t",
]

SENSOR_FILES = {
    "Accelerometer": "Accelerometer.csv",
    "Gyroscope": "Gyroscope.csv",
    "Gravity": "Gravity.csv",
    "Magnetometer": "Magnetometer.csv",
    "Orientation": "Orientation.csv",
}

ORIENTATION_AXES = ["pitch", "roll", "yaw", "azimuth"]
XYZ_AXES = ["x", "y", "z"]

GENERIC_SENSOR_COLUMN_MAPS: Dict[str, List[Dict[str, str]]] = {
    "Accelerometer": [
        {"x": "accel_x", "y": "accel_y", "z": "accel_z"},
        {"x": "accelerometer_x", "y": "accelerometer_y", "z": "accelerometer_z"},
    ],
    "Gyroscope": [
        {"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"},
        {"x": "gyroscope_x", "y": "gyroscope_y", "z": "gyroscope_z"},
    ],
    "Gravity": [
        {"x": "gravity_x", "y": "gravity_y", "z": "gravity_z"},
    ],
    "Magnetometer": [
        {"x": "mag_x", "y": "mag_y", "z": "mag_z"},
        {"x": "magnetometer_x", "y": "magnetometer_y", "z": "magnetometer_z"},
    ],
}

GENERIC_ORIENTATION_COLUMN_MAPS: List[Dict[str, str]] = [
    {"pitch": "pitch", "roll": "roll", "yaw": "yaw"},
    {"pitch": "orientation_pitch", "roll": "orientation_roll", "yaw": "orientation_yaw"},
]

BAND_LO_HZ = 0.10
BAND_HI_HZ = 0.80
FILTER_ORDER = 3
MAX_RESAMPLE_FS = 50.0

MIN_REPS_PER_SET = 3
TOP_CHANNELS = 7
MIN_CHANNEL_SUPPORT_FOR_SET = 2

# Gap used to split consecutive sets from one channel, scaled by estimated period.
SET_GAP_PERIOD_MULT = 2.0

# Jerk-based activity trimming (from prior notebook workflow).
JERK_ACTIVE_THRESH = 0.18
JERK_WIN_S = 1.5
DEAD_BUFFER_S = 0.8

# Cross-axis trust weighting (from prior notebook workflow).
CORR_TRUST_THRESHOLD = 0.50
LOW_TRUST_AXIS_WEIGHT = 0.55

# Template matching tunables:
# - cross-correlation for phase-tolerant waveform alignment
# - DTW for non-linear timing drift alignment
# - subsequence matching for template hits inside the full trace
TEMPLATE_RESAMPLE_N = 96
XCORR_MAX_LAG_FRAC = 0.22
DTW_BAND_FRAC = 0.16
SUBSEQ_STRIDE_S = 0.04
SUBSEQ_MIN_SCORE = 0.30
SUBSEQ_NEAR_CENTER_FRAC = 0.45

# Use only top-channel waveform/valley consensus (Graph 4 inputs) for rep/set counts.
# When True, skip peak-hybrid and cadence post-adjustments.
GRAPH4_ONLY_LOGIC = True

# Colab/Drive defaults. Leave as None unless you want a built-in default path.
DEFAULT_COLAB_SESSION: Optional[str] = None
DEFAULT_COLAB_ROOT: Optional[str] = None
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")


@dataclass
class ChannelSet:
    start: float
    end: float
    center: float
    rep_count: int
    interval_cv: float


@dataclass
class ChannelResult:
    sensor: str
    axis: str
    fs: float
    dominant_hz: float
    period_s: float
    periodicity: float
    snr: float
    score: float
    valleys_t: np.ndarray
    peaks_t: np.ndarray
    trace_t: np.ndarray
    trace_filt: np.ndarray
    sets: List[ChannelSet]
    axis_trust: float = 1.0
    template_xcorr: float = 0.0
    template_dtw: float = float("nan")
    template_subseq: float = 0.0
    template_match_score: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.sensor}:{self.axis}"

    @property
    def total_reps(self) -> int:
        return int(sum(s.rep_count for s in self.sets))



def _running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore

        _ = google.colab
        return True
    except Exception:
        return False


def _mount_google_drive(force_remount: bool = False) -> bool:
    if COLAB_DRIVE_ROOT.exists() and not force_remount:
        return True
    if not _running_in_colab():
        return False

    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=force_remount)
    except Exception as exc:
        print(f"Warning: failed to mount Google Drive: {exc}")
        return False

    return COLAB_DRIVE_ROOT.exists()


def _resolve_input_path(raw_path: str, prefer_colab_drive: bool = False) -> Path:
    raw = raw_path.strip()
    p = Path(raw).expanduser()

    candidates: List[Path] = []
    candidates.append(p)

    if raw.startswith("MyDrive/"):
        candidates.append(COLAB_DRIVE_ROOT / raw[len("MyDrive/") :])
    elif raw.startswith("drive/MyDrive/"):
        candidates.append(Path("/content") / raw)
    elif raw.startswith("/content/drive/MyDrive/"):
        candidates.append(Path(raw))
    elif prefer_colab_drive and not p.is_absolute():
        candidates.append(COLAB_DRIVE_ROOT / raw)

    # Preserve order and avoid duplicate checks.
    seen = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        k = str(c)
        if k in seen:
            continue
        seen.add(k)
        unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists():
            return c

    return unique_candidates[0]


def _resolve_output_path(raw_path: str) -> Path:
    raw = raw_path.strip()
    if raw.startswith("MyDrive/"):
        return COLAB_DRIVE_ROOT / raw[len("MyDrive/") :]
    if raw.startswith("drive/MyDrive/"):
        return Path("/content") / raw
    if raw.startswith("/content/drive/MyDrive/"):
        return Path(raw)
    return Path(raw).expanduser()


def _find_time_col(df: pd.DataFrame) -> Optional[str]:
    lm = {c.lower(): c for c in df.columns}
    for alias in TIME_ALIASES:
        if alias in df.columns:
            return alias
        if alias.lower() in lm:
            return lm[alias.lower()]
    return None


def _normalize_time_to_elapsed_seconds(t: Sequence[float]) -> np.ndarray:
    """
    Convert absolute timestamps to elapsed seconds.

    Handles common IMU exports that use Unix epoch values in nanoseconds,
    microseconds, milliseconds, or seconds.
    """
    arr = np.asarray(t, dtype=float).copy()
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr

    vals = arr[finite]
    abs_max = float(np.nanmax(np.abs(vals)))
    uniq = np.unique(vals)
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt_med = float(np.median(diffs)) if len(diffs) else float("nan")

    scale = 1.0
    if abs_max >= 1e17 or (np.isfinite(dt_med) and dt_med >= 1e7):
        scale = 1e9
    elif abs_max >= 1e14 or (np.isfinite(dt_med) and dt_med >= 1e4):
        scale = 1e6
    elif abs_max >= 1e11 or (np.isfinite(dt_med) and dt_med >= 10.0):
        scale = 1e3

    arr[finite] = arr[finite] / scale
    origin = float(arr[finite][0])
    arr[finite] = arr[finite] - origin
    return arr


def _extract_axes(df: pd.DataFrame, axis_map: Dict[str, str]) -> Optional[Dict[str, np.ndarray]]:
    axes: Dict[str, np.ndarray] = {}
    for axis, col in axis_map.items():
        if col not in df.columns:
            return None
        axes[axis] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return axes


def _infer_generic_sensor_label(path: Path, sensor_name: str) -> str:
    stem = path.stem.strip().replace(" ", "_")
    suffix = sensor_name.lower()
    if stem.lower().endswith(suffix):
        return stem
    return f"{stem}_{suffix}"


def _load_sensor_tables_for_session(session_dir: Path) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []

    for sensor_name, filename in SENSOR_FILES.items():
        fp = session_dir / filename
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        tcol = _find_time_col(df)
        if tcol is None:
            continue
        t = _normalize_time_to_elapsed_seconds(pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float))

        if sensor_name == "Orientation":
            axes = {axis: pd.to_numeric(df[axis], errors="coerce").to_numpy(dtype=float) for axis in ORIENTATION_AXES if axis in df.columns}
        else:
            axes = {axis: pd.to_numeric(df[axis], errors="coerce").to_numpy(dtype=float) for axis in XYZ_AXES if axis in df.columns}

        if axes:
            tables.append({"sensor": sensor_name, "time": t, "axes": axes, "path": fp})

    for fp in sorted(session_dir.glob("*.csv")):
        if fp.name in SENSOR_FILES.values():
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        tcol = _find_time_col(df)
        if tcol is None:
            continue
        t = _normalize_time_to_elapsed_seconds(pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float))

        for sensor_name, maps in GENERIC_SENSOR_COLUMN_MAPS.items():
            axes = None
            for axis_map in maps:
                axes = _extract_axes(df, axis_map)
                if axes is not None:
                    break
            if axes:
                tables.append(
                    {
                        "sensor": _infer_generic_sensor_label(fp, sensor_name),
                        "time": t,
                        "axes": axes,
                        "path": fp,
                    }
                )

        orient_axes = None
        for axis_map in GENERIC_ORIENTATION_COLUMN_MAPS:
            orient_axes = _extract_axes(df, axis_map)
            if orient_axes is not None:
                break
        if orient_axes:
            tables.append(
                {
                    "sensor": _infer_generic_sensor_label(fp, "Orientation"),
                    "time": t,
                    "axes": orient_axes,
                    "path": fp,
                }
            )

    return tables


def _bandpass(sig: np.ndarray, fs: float, lo_hz: float = BAND_LO_HZ, hi_hz: float = BAND_HI_HZ) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(lo_hz / nyq, 1e-4)
    hi = min(hi_hz / nyq, 0.999)
    if lo >= hi:
        b, a = butter(FILTER_ORDER, hi, btype="low")
    else:
        b, a = butter(FILTER_ORDER, [lo, hi], btype="band")
    return filtfilt(b, a, sig)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = float(np.std(x))
    if s < 1e-9:
        return x - float(np.mean(x))
    return (x - float(np.mean(x))) / s


def _rolling_rms(x: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    n = max(3, int(round(float(win_s) * float(fs))))
    if n % 2 == 0:
        n += 1
    return np.sqrt(np.maximum(0.0, uniform_filter1d(x * x, size=n, mode="nearest")))


def _active_window_from_jerk(
    t: np.ndarray,
    sig: np.ndarray,
    fs: float,
) -> Tuple[float, float]:
    """
    Estimate active movement window using jerk envelope.

    Returns (start_s, end_s) bounds in time coordinates.
    """
    tt = np.asarray(t, dtype=float)
    xx = np.asarray(sig, dtype=float)
    if len(tt) == 0:
        return 0.0, 0.0
    if len(tt) < 10 or not np.isfinite(fs) or fs <= 0:
        return float(tt[0]), float(tt[-1])

    jerk = np.gradient(xx, 1.0 / float(fs))
    env = _rolling_rms(np.abs(jerk), fs, JERK_WIN_S)
    peak = float(np.nanmax(env)) if len(env) else 0.0
    if not np.isfinite(peak) or peak <= 1e-9:
        return float(tt[0]), float(tt[-1])

    thr = float(JERK_ACTIVE_THRESH * peak)
    active = env > thr
    if not np.any(active):
        return float(tt[0]), float(tt[-1])

    idx = np.flatnonzero(active)
    start = float(tt[idx[0]] - DEAD_BUFFER_S)
    end = float(tt[idx[-1]] + DEAD_BUFFER_S)
    return float(max(start, float(tt[0]))), float(min(end, float(tt[-1])))


def _downsample_if_needed(t: np.ndarray, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
    if fs <= MAX_RESAMPLE_FS:
        return t, x, fs
    step = int(math.ceil(fs / MAX_RESAMPLE_FS))
    t2 = t[::step]
    x2 = x[::step]
    if len(t2) < 4:
        return t, x, fs
    fs2 = 1.0 / float(np.median(np.diff(t2)))
    return t2, x2, fs2


def _ensure_monotonic_time(t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Drop non-finite and repeated/non-increasing time rows.
    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]
    if len(t) < 4:
        return t, x

    order = np.argsort(t)
    t = t[order]
    x = x[order]
    dt = np.diff(t)
    keep = np.r_[True, dt > 0]
    return t[keep], x[keep]


def _split_into_sets(valley_t: np.ndarray, period_s: float) -> List[np.ndarray]:
    if len(valley_t) == 0:
        return []
    if len(valley_t) == 1:
        return [valley_t]

    gap_thr = max(2.5, SET_GAP_PERIOD_MULT * period_s)
    gaps = np.diff(valley_t)
    split_idx = np.where(gaps > gap_thr)[0]

    groups: List[np.ndarray] = []
    st = 0
    for idx in split_idx:
        groups.append(valley_t[st : idx + 1])
        st = idx + 1
    groups.append(valley_t[st:])
    return groups


def _trim_set_edges(valley_t: np.ndarray, period_s: float) -> np.ndarray:
    """
    Trim boundary valleys that look like setup/rack transitions rather than full reps.

    Uses only edge interval anomalies relative to internal cadence, so stable interior
    reps remain untouched.
    """
    v = np.asarray(valley_t, dtype=float)
    if len(v) < 4:
        return v

    while True:
        changed = False
        if len(v) < 4:
            break

        d = np.diff(v)
        core = d[1:-1] if len(d) > 2 else d
        if len(core) == 0:
            break

        ref = float(np.median(core))
        # Allow moderate cadence variation but reject large boundary jumps.
        high = max(1.35 * ref, ref + 0.9, 1.15 * period_s)
        low = 0.55 * ref

        d = np.diff(v)
        if len(d) >= 1 and (d[0] > high or d[0] < low):
            v = v[1:]
            changed = True

        if len(v) >= 4:
            d = np.diff(v)
            if len(d) >= 1 and (d[-1] > high or d[-1] < low):
                v = v[:-1]
                changed = True

        if not changed:
            break

    return v


def _interval_cv(times: np.ndarray) -> float:
    if len(times) < 3:
        return 1.0
    d = np.diff(times)
    m = float(np.mean(d))
    if m <= 1e-9:
        return 1.0
    return float(np.std(d) / m)


def _dedupe_sorted_times(times: Sequence[float], min_sep_s: float) -> np.ndarray:
    arr = np.asarray([float(v) for v in times if np.isfinite(float(v))], dtype=float)
    arr = np.unique(np.sort(arr))
    if len(arr) == 0:
        return np.array([], dtype=float)

    out: List[float] = [float(arr[0])]
    for v in arr[1:]:
        cur = float(v)
        if (cur - out[-1]) >= float(min_sep_s):
            out.append(cur)
    return np.asarray(out, dtype=float)


def _resample_to_length(x: np.ndarray, n_target: int) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    n = int(max(2, n_target))
    if len(xx) == 0:
        return np.zeros(n, dtype=float)
    if len(xx) == 1:
        return np.full(n, float(xx[0]), dtype=float)
    if len(xx) == n:
        return xx.astype(float, copy=True)

    old = np.linspace(0.0, 1.0, len(xx), dtype=float)
    new = np.linspace(0.0, 1.0, n, dtype=float)
    return np.interp(new, old, xx).astype(float)


def _best_lagged_cross_correlation(
    a: np.ndarray,
    b: np.ndarray,
    max_lag_frac: float = XCORR_MAX_LAG_FRAC,
) -> Tuple[float, int]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    n = int(min(len(aa), len(bb)))
    if n < 6:
        return 0.0, 0

    aa = aa[:n]
    bb = bb[:n]
    max_lag = int(max(1, round(float(max_lag_frac) * n)))

    best_r = -1.0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = aa[-lag:]
            y = bb[: n + lag]
        elif lag > 0:
            x = aa[: n - lag]
            y = bb[lag:]
        else:
            x = aa
            y = bb

        if len(x) < 6:
            continue
        sx = float(np.std(x))
        sy = float(np.std(y))
        if sx <= 1e-9 or sy <= 1e-9:
            continue

        xz = (x - float(np.mean(x))) / sx
        yz = (y - float(np.mean(y))) / sy
        r = float(np.dot(xz, yz) / len(xz))
        if r > best_r:
            best_r = r
            best_lag = int(lag)

    if not np.isfinite(best_r):
        return 0.0, 0
    return float(np.clip(best_r, -1.0, 1.0)), int(best_lag)


def _dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    band_frac: float = DTW_BAND_FRAC,
) -> float:
    aa = _zscore(np.asarray(a, dtype=float))
    bb = _zscore(np.asarray(b, dtype=float))
    n = int(len(aa))
    m = int(len(bb))
    if n < 2 or m < 2:
        return float("inf")

    band = int(max(abs(n - m), round(float(band_frac) * max(n, m))))
    inf = float("inf")
    prev = np.full(m + 1, inf, dtype=float)
    prev[0] = 0.0

    for i in range(1, n + 1):
        cur = np.full(m + 1, inf, dtype=float)
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = abs(float(aa[i - 1]) - float(bb[j - 1]))
            cur[j] = cost + min(cur[j - 1], prev[j], prev[j - 1])
        prev = cur

    dist = float(prev[m])
    if not np.isfinite(dist):
        return float("inf")
    return float(dist / max(n, m))


def _subsequence_match_scores(
    trace: np.ndarray,
    template: np.ndarray,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(trace, dtype=float)
    tpl = np.asarray(template, dtype=float)
    n = int(len(x))
    m = int(len(tpl))
    if n < m or m < 6:
        return np.array([], dtype=int), np.array([], dtype=float)

    step = int(max(1, stride))
    tpl_std = float(np.std(tpl))
    if tpl_std <= 1e-9:
        return np.array([], dtype=int), np.array([], dtype=float)
    tpl_z = (tpl - float(np.mean(tpl))) / tpl_std

    starts = np.arange(0, n - m + 1, step, dtype=int)
    centers = starts + (m // 2)
    scores = np.zeros(len(starts), dtype=float)

    for i, s in enumerate(starts):
        w = x[s : s + m]
        sw = float(np.std(w))
        if sw <= 1e-9:
            scores[i] = 0.0
            continue
        wz = (w - float(np.mean(w))) / sw
        scores[i] = float(np.dot(wz, tpl_z) / len(tpl_z))

    return centers.astype(int), np.asarray(scores, dtype=float)


def _template_matching_summary(
    t: np.ndarray,
    filt: np.ndarray,
    valley_t: Sequence[float],
    period_s: float,
    fs: float,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "template_xcorr": 0.0,
        "template_dtw": float("nan"),
        "template_subseq": 0.0,
        "template_match_score": 0.0,
        "subsequence_t": np.array([], dtype=float),
    }

    tt = np.asarray(t, dtype=float)
    ff = np.asarray(filt, dtype=float)
    vv = _dedupe_sorted_times(valley_t, min_sep_s=max(0.15 * float(period_s), 1e-3))
    if len(tt) < 40 or len(ff) != len(tt):
        return summary

    segments: List[np.ndarray] = []
    seg_centers: List[float] = []
    min_dur = 0.45 * float(period_s)
    max_dur = 1.90 * float(period_s)

    for i in range(len(vv) - 1):
        lo_t = float(vv[i])
        hi_t = float(vv[i + 1])
        dur = float(hi_t - lo_t)
        if not (min_dur <= dur <= max_dur):
            continue

        i0 = int(np.searchsorted(tt, lo_t, side="left"))
        i1 = int(np.searchsorted(tt, hi_t, side="right")) - 1
        if i1 <= (i0 + 6):
            continue
        seg = ff[i0 : i1 + 1]
        if len(seg) < 10:
            continue

        segments.append(_resample_to_length(seg, TEMPLATE_RESAMPLE_N))
        seg_centers.append(float(0.5 * (lo_t + hi_t)))

    if not segments:
        return summary

    seg_arr = np.vstack(segments)
    template = np.median(seg_arr, axis=0)

    have_pairwise = len(seg_arr) >= 2
    if have_pairwise:
        xcorr_vals: List[float] = []
        dtw_vals: List[float] = []
        for seg in seg_arr:
            r, _ = _best_lagged_cross_correlation(seg, template, max_lag_frac=XCORR_MAX_LAG_FRAC)
            xcorr_vals.append(float(max(r, 0.0)))
            d = _dtw_distance(seg, template, band_frac=DTW_BAND_FRAC)
            if np.isfinite(d):
                dtw_vals.append(float(max(d, 0.0)))

        if xcorr_vals:
            summary["template_xcorr"] = float(np.clip(np.median(np.asarray(xcorr_vals, dtype=float)), 0.0, 1.0))
        if dtw_vals:
            summary["template_dtw"] = float(np.median(np.asarray(dtw_vals, dtype=float)))

    stride = int(max(1, round(float(SUBSEQ_STRIDE_S) * float(fs))))
    centers_idx, scores = _subsequence_match_scores(ff, template, stride=stride)
    if len(scores):
        score_thr = max(float(SUBSEQ_MIN_SCORE), float(np.quantile(scores, 0.78)))
        sep_idx = int(max(1, round((0.55 * float(period_s) * float(fs)) / max(stride, 1))))
        p_idx, _ = find_peaks(scores, distance=sep_idx, height=score_thr, prominence=0.05)
        if len(p_idx):
            match_idx = centers_idx[p_idx]
            match_idx = np.clip(match_idx.astype(int), 0, len(tt) - 1)
            subseq_t = tt[match_idx]
            subseq_t = _dedupe_sorted_times(subseq_t, min_sep_s=0.45 * float(period_s))
            summary["subsequence_t"] = subseq_t

            match_scores = scores[p_idx]
            if len(seg_centers):
                centers_t = tt[np.clip(centers_idx.astype(int), 0, len(tt) - 1)]
                near_tol = float(max(0.16, SUBSEQ_NEAR_CENTER_FRAC * float(period_s)))
                near_scores: List[float] = []
                for c0 in seg_centers:
                    near = scores[np.abs(centers_t - float(c0)) <= near_tol]
                    if len(near):
                        near_scores.append(float(np.max(near)))
                if near_scores:
                    summary["template_subseq"] = float(
                        np.clip(np.median(np.asarray(near_scores, dtype=float)), 0.0, 1.0)
                    )
                else:
                    summary["template_subseq"] = float(
                        np.clip(np.median(np.asarray(match_scores, dtype=float)), 0.0, 1.0)
                    )
            else:
                summary["template_subseq"] = float(
                    np.clip(np.median(np.asarray(match_scores, dtype=float)), 0.0, 1.0)
                )

    xcorr = float(summary["template_xcorr"])
    dtw = float(summary["template_dtw"])
    dtw_sim = float(math.exp(-max(dtw, 0.0))) if have_pairwise else 0.0
    subseq = float(summary["template_subseq"])
    summary["template_match_score"] = float(np.clip(0.45 * xcorr + 0.35 * dtw_sim + 0.20 * subseq, 0.0, 1.0))
    return summary


def _detect_channel(t_raw: np.ndarray, x_raw: np.ndarray, sensor: str, axis: str) -> Optional[ChannelResult]:
    t, x = _ensure_monotonic_time(np.asarray(t_raw, dtype=float), np.asarray(x_raw, dtype=float))
    if len(t) < 300:
        return None

    fs = 1.0 / float(np.median(np.diff(t)))
    t, x, fs = _downsample_if_needed(t, x, fs)
    if len(t) < 180:
        return None

    x = _zscore(x)

    try:
        filt = _bandpass(x, fs)
    except Exception:
        return None

    # Dominant periodic component in expected rep band.
    nperseg = min(len(filt), max(256, int(fs * 30)))
    f, pxx = welch(filt, fs=fs, nperseg=nperseg)
    mask = (f >= BAND_LO_HZ) & (f <= BAND_HI_HZ)
    if int(np.sum(mask)) < 5:
        return None

    f_band = f[mask]
    p_band = pxx[mask]
    pk_idx = int(np.argmax(p_band))
    dominant_hz = float(f_band[pk_idx])
    period_s = 1.0 / max(dominant_hz, 0.12)

    # Cadence-adaptive extrema detection settings.
    min_sep_s = 0.55 * period_s
    min_dist = max(1, int(fs * min_sep_s))
    prominence = max(0.18 * float(np.std(filt)), 0.08)

    env = np.abs(hilbert(filt))
    env_thr = float(np.quantile(env, 0.35))
    min_pair_sep = 0.45 * period_s

    def _extract_extrema(prefer_lower: bool) -> np.ndarray:
        sig = -filt if prefer_lower else filt
        idx, props = find_peaks(sig, distance=min_dist, prominence=prominence)
        if len(idx) == 0:
            return np.array([], dtype=int)

        keep_env = env[idx] >= env_thr
        idx = idx[keep_env]
        prom = props["prominences"][keep_env]
        if len(idx) == 0:
            return np.array([], dtype=int)

        prom_thr = 0.65 * float(np.median(prom)) if len(prom) else 0.0
        keep_prom = prom >= prom_thr
        idx = idx[keep_prom]
        if len(idx) == 0:
            return np.array([], dtype=int)

        dedup: List[int] = []
        for cur in idx:
            cur_i = int(cur)
            if not dedup:
                dedup.append(cur_i)
                continue
            last = dedup[-1]
            if (t[cur_i] - t[last]) < min_pair_sep:
                better = (filt[cur_i] < filt[last]) if prefer_lower else (filt[cur_i] > filt[last])
                if better:
                    dedup[-1] = cur_i
            else:
                dedup.append(cur_i)

        return np.asarray(dedup, dtype=int)

    valley_idx = _extract_extrema(prefer_lower=True)
    if len(valley_idx) == 0:
        return None

    peak_idx = _extract_extrema(prefer_lower=False)

    valley_t_all = t[valley_idx]
    valley_t = valley_t_all.copy()
    peak_t = t[peak_idx] if len(peak_idx) else np.array([], dtype=float)

    # Trim setup/rack dead space using jerk activity envelope.
    act_lo, act_hi = _active_window_from_jerk(t, filt, fs)
    valley_t = valley_t[(valley_t >= act_lo) & (valley_t <= act_hi)]
    if len(peak_t):
        peak_t = peak_t[(peak_t >= act_lo) & (peak_t <= act_hi)]

    template_summary = _template_matching_summary(t, filt, valley_t_all, period_s=period_s, fs=fs)
    subseq_t = np.asarray(template_summary.get("subsequence_t", np.array([], dtype=float)), dtype=float)
    if len(subseq_t):
        subseq_t = subseq_t[(subseq_t >= act_lo) & (subseq_t <= act_hi)]
        subseq_t = _dedupe_sorted_times(subseq_t, min_sep_s=min_pair_sep)

    if len(valley_t) == 0 and len(subseq_t):
        valley_t = subseq_t
    elif (
        len(valley_t) < MIN_REPS_PER_SET
        and len(subseq_t) >= MIN_REPS_PER_SET
        and float(template_summary.get("template_subseq", 0.0)) >= 0.42
    ):
        valley_t = _dedupe_sorted_times(np.r_[valley_t, subseq_t], min_sep_s=min_pair_sep)

    if len(valley_t) == 0:
        return None

    # Build per-channel set candidates from valley anchors.
    set_groups = _split_into_sets(valley_t, period_s)
    channel_sets: List[ChannelSet] = []
    for grp in set_groups:
        grp = _trim_set_edges(grp, period_s)
        if len(grp) < MIN_REPS_PER_SET:
            continue
        channel_sets.append(
            ChannelSet(
                start=float(grp[0]),
                end=float(grp[-1]),
                center=float(0.5 * (grp[0] + grp[-1])),
                rep_count=int(len(grp)),
                interval_cv=_interval_cv(grp),
            )
        )

    # Quality score: periodicity * spectral concentration.
    lag = int(round(period_s * fs))
    periodicity = 0.0
    if 2 <= lag < (len(filt) - 2):
        a = filt[:-lag]
        b = filt[lag:]
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            periodicity = float(np.corrcoef(a, b)[0, 1])

    snr = float(p_band[pk_idx] / (np.median(p_band) + 1e-12))
    base_score = max(periodicity, 0.0) * math.log1p(max(snr, 0.0))
    template_match_score = float(template_summary.get("template_match_score", 0.0))
    score = base_score * (1.0 + 0.30 * template_match_score)

    return ChannelResult(
        sensor=sensor,
        axis=axis,
        fs=float(fs),
        dominant_hz=dominant_hz,
        period_s=period_s,
        periodicity=periodicity,
        snr=snr,
        score=score,
        valleys_t=valley_t,
        peaks_t=peak_t,
        trace_t=t,
        trace_filt=filt,
        sets=channel_sets,
        template_xcorr=float(template_summary.get("template_xcorr", 0.0)),
        template_dtw=float(template_summary.get("template_dtw", 0.0)),
        template_subseq=float(template_summary.get("template_subseq", 0.0)),
        template_match_score=template_match_score,
    )


def _load_channels_for_session(session_dir: Path) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    channels: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

    for table in _load_sensor_tables_for_session(session_dir):
        sensor_name = str(table["sensor"])
        t = np.asarray(table["time"], dtype=float)
        axes = dict(table["axes"])
        for axis, x in axes.items():
            channels.append((sensor_name, str(axis), t, np.asarray(x, dtype=float)))

    return channels


def _compute_axis_trust(
    detected: Sequence[ChannelResult],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute per-channel trust from cross-axis correlations within each sensor.

    Returns:
      - trust score in [0,1] per channel name
      - trust weight multiplier used during ranking/aggregation
    """
    trust_scores: Dict[str, float] = {c.name: 1.0 for c in detected}
    trust_weights: Dict[str, float] = {c.name: 1.0 for c in detected}

    by_sensor: Dict[str, List[ChannelResult]] = {}
    for c in detected:
        if str(c.axis).lower() in {"x", "y", "z"}:
            by_sensor.setdefault(c.sensor, []).append(c)

    for sensor_channels in by_sensor.values():
        if len(sensor_channels) < 2:
            continue

        n = len(sensor_channels)
        corr = np.eye(n, dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                xi = np.asarray(sensor_channels[i].trace_filt, dtype=float)
                xj = np.asarray(sensor_channels[j].trace_filt, dtype=float)
                L = min(len(xi), len(xj))
                if L < 60:
                    r = 0.0
                else:
                    ai = xi[:L]
                    bj = xj[:L]
                    if np.std(ai) <= 1e-9 or np.std(bj) <= 1e-9:
                        r = 0.0
                    else:
                        r = float(abs(np.corrcoef(ai, bj)[0, 1]))
                        if not np.isfinite(r):
                            r = 0.0
                corr[i, j] = r
                corr[j, i] = r

        for i, ch in enumerate(sensor_channels):
            others = [float(corr[i, j]) for j in range(n) if j != i and np.isfinite(corr[i, j])]
            if not others:
                continue
            trust = float(np.clip(np.mean(np.asarray(others, dtype=float)), 0.0, 1.0))
            trust_scores[ch.name] = trust
            if trust >= CORR_TRUST_THRESHOLD:
                w = 1.0
            else:
                ratio = trust / max(CORR_TRUST_THRESHOLD, 1e-6)
                w = float(np.clip(LOW_TRUST_AXIS_WEIGHT * ratio, 0.30, 0.95))
            trust_weights[ch.name] = w

    return trust_scores, trust_weights


def _cluster_set_candidates(
    set_candidates: List[Tuple[ChannelResult, ChannelSet]],
    window_s: float,
    n_selected_channels: int,
) -> List[Dict[str, Any]]:
    if not set_candidates:
        return []

    set_candidates = sorted(set_candidates, key=lambda item: item[1].center)

    groups: List[List[Tuple[ChannelResult, ChannelSet]]] = []
    group: List[Tuple[ChannelResult, ChannelSet]] = [set_candidates[0]]

    def group_center(g: List[Tuple[ChannelResult, ChannelSet]]) -> float:
        return float(np.median([s.center for _, s in g]))

    for cand in set_candidates[1:]:
        c_center = cand[1].center
        if abs(c_center - group_center(group)) <= window_s:
            group.append(cand)
        else:
            groups.append(group)
            group = [cand]
    groups.append(group)

    consensus_sets: List[Dict[str, Any]] = []
    for g in groups:
        # Ensure unique channel support inside each cluster.
        unique_channels = {}
        for cr, cs in g:
            unique_channels.setdefault(cr.name, []).append((cr, cs))

        # Keep one best candidate per channel (highest channel score).
        members: List[Tuple[ChannelResult, ChannelSet]] = []
        for channel_entries in unique_channels.values():
            best = max(channel_entries, key=lambda pair: pair[0].score)
            members.append(best)

        if len(members) < MIN_CHANNEL_SUPPORT_FOR_SET:
            continue

        rep_counts = np.array([cs.rep_count for _, cs in members], dtype=float)
        starts = np.array([cs.start for _, cs in members], dtype=float)
        ends = np.array([cs.end for _, cs in members], dtype=float)
        cvs = np.array([cs.interval_cv for _, cs in members], dtype=float)

        support_ratio = len(members) / max(1, n_selected_channels)
        cv_med = float(np.median(cvs)) if len(cvs) else 1.0
        cv_score = 1.0 - min(cv_med / 0.35, 1.0)
        count_disp = float(np.std(rep_counts))
        count_score = 1.0 - min(count_disp / 2.0, 1.0)

        quality = 0.5 * support_ratio + 0.35 * cv_score + 0.15 * count_score
        rep_count = int(round(float(np.median(rep_counts))))

        proper = bool((quality >= 0.58) and (rep_count >= 3))

        consensus_sets.append(
            {
                "start_s": float(np.median(starts)),
                "end_s": float(np.median(ends)),
                "rep_count": rep_count,
                "channel_support": int(len(members)),
                "support_ratio": float(support_ratio),
                "interval_cv": cv_med,
                "quality_score": float(quality),
                "proper": proper,
                "supporting_channels": sorted([cr.name for cr, _ in members]),
            }
        )

    return sorted(consensus_sets, key=lambda d: d["start_s"])


def _apply_turning_point_start_consensus(
    consensus_sets: List[Dict[str, Any]],
    selected_channels: Sequence[ChannelResult],
) -> None:
    """
    Move each set start to the first peak/valley cluster that reaches channel consensus.
    """
    if not consensus_sets or not selected_channels:
        return

    channel_map = {c.name: c for c in selected_channels}

    def _cluster_events(
        events: List[Tuple[float, str, str]],
        tol_s: float,
    ) -> List[Dict[str, Any]]:
        if not events:
            return []
        events = sorted(events, key=lambda x: float(x[0]))
        groups: List[List[Tuple[float, str, str]]] = [[events[0]]]
        for ev in events[1:]:
            g = groups[-1]
            ctime = float(np.median(np.asarray([e[0] for e in g], dtype=float)))
            if abs(float(ev[0]) - ctime) <= tol_s:
                g.append(ev)
            else:
                groups.append([ev])

        out: List[Dict[str, Any]] = []
        for g in groups:
            times = np.asarray([float(e[0]) for e in g], dtype=float)
            channels = {str(e[1]) for e in g}
            peak_votes = int(sum(1 for e in g if str(e[2]) == "peak"))
            valley_votes = int(len(g) - peak_votes)
            anchor_type = "peak" if peak_votes >= valley_votes else "valley"
            out.append(
                {
                    "time": float(np.median(times)),
                    "support": int(len(channels)),
                    "anchor_type": anchor_type,
                }
            )
        return out

    for s in consensus_sets:
        start_s = float(s.get("start_s", np.nan))
        end_s = float(s.get("end_s", np.nan))
        if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            continue

        support_names = [n for n in (s.get("supporting_channels") or []) if n in channel_map]
        support_channels = [channel_map[n] for n in support_names] if support_names else list(selected_channels)
        if not support_channels:
            continue

        periods = [float(c.period_s) for c in support_channels if np.isfinite(c.period_s) and c.period_s > 0]
        if not periods:
            continue
        period_s = float(np.median(np.asarray(periods, dtype=float)))

        search_lo = float(start_s - 1.15 * period_s)
        search_hi = float(start_s + 1.15 * period_s)
        events: List[Tuple[float, str, str]] = []
        for c in support_channels:
            p = np.asarray(c.peaks_t, dtype=float)
            p = p[np.isfinite(p)]
            p = p[(p >= search_lo) & (p <= search_hi)]
            for t0 in p:
                events.append((float(t0), c.name, "peak"))

            v = np.asarray(c.valleys_t, dtype=float)
            v = v[np.isfinite(v)]
            v = v[(v >= search_lo) & (v <= search_hi)]
            for t0 in v:
                events.append((float(t0), c.name, "valley"))

        if not events:
            continue

        tol_s = max(0.16, 0.26 * period_s)
        clusters = _cluster_events(events, tol_s=tol_s)
        if not clusters:
            continue

        min_support = max(MIN_CHANNEL_SUPPORT_FOR_SET, int(math.ceil(0.45 * len(support_channels))))
        eligible = [c for c in clusters if int(c["support"]) >= min_support]
        if eligible:
            chosen = min(eligible, key=lambda x: float(x["time"]))
        else:
            best_sup = max(int(c["support"]) for c in clusters)
            best = [c for c in clusters if int(c["support"]) == best_sup]
            chosen = min(best, key=lambda x: float(x["time"]))

        new_start = float(chosen["time"])
        if not (search_lo <= new_start <= search_hi):
            continue
        if new_start >= (end_s - 0.25 * period_s):
            continue

        s["start_s_raw"] = float(start_s)
        s["start_s"] = float(new_start)
        s["start_anchor"] = str(chosen["anchor_type"])
        s["start_anchor_support"] = int(chosen["support"])


def _apply_half_period_phase_shift_rep_counts(
    consensus_sets: List[Dict[str, Any]],
    selected_channels: Sequence[ChannelResult],
) -> None:
    """
    Apply a half-period phase correction when duration/period suggests one missed rep.
    """
    if not consensus_sets or not selected_channels:
        return

    channel_map = {c.name: c for c in selected_channels}

    for s in consensus_sets:
        current = int(s.get("rep_count", 0))
        start_s = float(s.get("start_s", np.nan))
        end_s = float(s.get("end_s", np.nan))
        if current <= 0 or not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            continue

        support_names = [n for n in (s.get("supporting_channels") or []) if n in channel_map]
        support_channels = [channel_map[n] for n in support_names] if support_names else list(selected_channels)
        if not support_channels:
            continue

        periods = [float(c.period_s) for c in support_channels if np.isfinite(c.period_s) and c.period_s > 0]
        if not periods:
            continue
        period_s = float(np.median(np.asarray(periods, dtype=float)))
        half = 0.5 * period_s

        duration = float(end_s - start_s)
        duration_based = int(math.ceil(duration / max(period_s, 1e-6))) + 1
        anchor_type = str(s.get("start_anchor", ""))
        if anchor_type == "peak" and duration_based >= (current + 1):
            target = int(current + 1)
        else:
            target = int(current)
        if target <= current:
            continue

        s["rep_count_raw_phase"] = int(current)
        s["rep_count"] = int(target)
        prior_method = str(s.get("rep_count_method", "graph4_waveform_valley_consensus"))
        if "half_period_shift" not in prior_method:
            s["rep_count_method"] = f"{prior_method}+half_period_shift"
        s["phase_shift_half_period_s"] = float(half)
        s["phase_shift_support_ratio"] = 1.0
        s["proper"] = bool((float(s.get("quality_score", 0.0)) >= 0.58) and (int(target) >= 3))


def _weighted_median_int(values: Sequence[int], weights: Sequence[float]) -> int:
    if not values:
        return 0

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(w) != len(v):
        w = np.ones_like(v)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if float(np.sum(w)) <= 1e-9:
        w = np.ones_like(v)

    order = np.argsort(v)
    v = v[order]
    w = w[order]
    csum = np.cumsum(w)
    cutoff = 0.5 * float(np.sum(w))
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return int(round(float(v[idx])))


def _count_midpoint_windows(
    events_t: np.ndarray,
    opposite_events_t: np.ndarray,
    win_lo: float,
    win_hi: float,
) -> Tuple[int, int]:
    """
    Count midpoint windows built from consecutive events.

    For each event, define a repetition window by half-distance to neighboring events.
    Return: (number of windows overlapping [win_lo, win_hi],
             number of those windows containing at least one opposite event).
    """
    events = np.asarray(events_t, dtype=float)
    opp = np.asarray(opposite_events_t, dtype=float)

    events = events[np.isfinite(events)]
    opp = opp[np.isfinite(opp)]
    events = np.unique(np.sort(events))
    opp = np.sort(opp)

    if len(events) == 0:
        return 0, 0

    if len(events) == 1:
        inside = int(win_lo <= float(events[0]) <= win_hi)
        has_opp = inside and int(np.any((opp >= win_lo) & (opp <= win_hi)))
        return inside, int(has_opp)

    n_windows = 0
    n_with_opp = 0

    for i, center in enumerate(events):
        if i == 0:
            left = float(center - 0.5 * (events[1] - events[0]))
        else:
            left = float(0.5 * (events[i - 1] + events[i]))

        if i == (len(events) - 1):
            right = float(center + 0.5 * (events[-1] - events[-2]))
        else:
            right = float(0.5 * (events[i] + events[i + 1]))

        if max(left, win_lo) <= min(right, win_hi):
            n_windows += 1
            if np.any((opp >= left) & (opp <= right)):
                n_with_opp += 1

    return int(n_windows), int(n_with_opp)


def _estimate_rep_spacing(
    centers: Sequence[float],
    start_s: float,
    end_s: float,
    target_reps: int,
    period_s: float,
) -> float:
    vals: List[float] = []

    arr = np.asarray([float(c) for c in centers if np.isfinite(c)], dtype=float)
    arr = np.unique(np.sort(arr))
    if len(arr) >= 2:
        d = np.diff(arr)
        d = d[np.isfinite(d) & (d > 1e-6)]
        if len(d):
            if len(d) >= 3:
                q_lo, q_hi = np.quantile(d, [0.2, 0.8])
                d_trim = d[(d >= q_lo) & (d <= q_hi)]
                if len(d_trim) == 0:
                    d_trim = d
            else:
                d_trim = d
            vals.append(float(np.median(d_trim)))

    if target_reps > 1 and np.isfinite(end_s) and np.isfinite(start_s) and end_s > start_s:
        vals.append(float((end_s - start_s) / float(target_reps - 1)))

    if np.isfinite(period_s) and period_s > 0:
        vals.append(float(period_s))

    if not vals:
        return max(float(period_s) if (np.isfinite(period_s) and period_s > 0) else 1.0, 1e-6)

    step = float(np.median(np.asarray(vals, dtype=float)))
    if np.isfinite(period_s) and period_s > 0:
        step = float(np.clip(step, 0.55 * period_s, 1.65 * period_s))
    return max(step, 1e-6)


def _midpoint_windows_from_centers(
    centers: Sequence[float],
    period_s: float,
) -> List[Tuple[float, float, float]]:
    arr = np.asarray([float(c) for c in centers if np.isfinite(c)], dtype=float)
    arr = np.unique(np.sort(arr))
    if len(arr) == 0:
        return []
    if len(arr) == 1:
        c = float(arr[0])
        return [(c - 0.5 * period_s, c + 0.5 * period_s, c)]

    out: List[Tuple[float, float, float]] = []
    for i, c in enumerate(arr):
        if i == 0:
            left = float(c - 0.5 * (arr[1] - arr[0]))
        else:
            left = float(0.5 * (arr[i - 1] + arr[i]))

        if i == (len(arr) - 1):
            right = float(c + 0.5 * (arr[-1] - arr[-2]))
        else:
            right = float(0.5 * (arr[i] + arr[i + 1]))

        out.append((left, right, float(c)))
    return out


def _valley_anchored_windows_from_centers(
    centers: Sequence[float],
    valleys: Sequence[float],
    period_s: float,
) -> List[Tuple[float, float, float]]:
    """
    Build rep windows from centers, but anchor boundaries to nearby valleys
    whenever possible. This keeps start/stop lines aligned to true turning points.
    """
    arr = np.asarray([float(c) for c in centers if np.isfinite(c)], dtype=float)
    arr = np.unique(np.sort(arr))
    if len(arr) == 0:
        return []

    vals = np.asarray([float(v) for v in valleys if np.isfinite(v)], dtype=float)
    vals = np.unique(np.sort(vals))
    if len(vals) == 0:
        return _midpoint_windows_from_centers(arr, period_s)

    if len(arr) == 1:
        c = float(arr[0])
        left_cand = vals[vals < c]
        right_cand = vals[vals > c]
        left = float(left_cand[-1]) if len(left_cand) else float(c - 0.5 * period_s)
        right = float(right_cand[0]) if len(right_cand) else float(c + 0.5 * period_s)
        if right <= left:
            right = float(left + max(0.40 * period_s, 1e-3))
        return [(left, right, c)]

    internal_bounds: List[float] = []
    for i in range(len(arr) - 1):
        lo = float(arr[i])
        hi = float(arr[i + 1])
        mid = float(0.5 * (lo + hi))
        cand = vals[(vals > lo) & (vals < hi)]
        if len(cand):
            b = float(cand[int(np.argmin(np.abs(cand - mid)))])
        else:
            b = mid
        internal_bounds.append(b)

    left_cand = vals[vals < arr[0]]
    right_cand = vals[vals > arr[-1]]
    left = float(left_cand[-1]) if len(left_cand) else float(arr[0] - 0.5 * (arr[1] - arr[0]))
    right = float(right_cand[0]) if len(right_cand) else float(arr[-1] + 0.5 * (arr[-1] - arr[-2]))

    bounds = [left] + internal_bounds + [right]
    min_w = max(0.08 * period_s, 1e-3)

    for i in range(1, len(bounds)):
        if bounds[i] <= (bounds[i - 1] + min_w):
            bounds[i] = float(bounds[i - 1] + min_w)

    for i in range(1, len(bounds) - 1):
        lo = float(arr[i - 1] + 0.05 * period_s)
        hi = float(arr[i] - 0.05 * period_s)
        if hi <= lo:
            hi = float(lo + min_w)
        bounds[i] = float(np.clip(bounds[i], lo, hi))

    for i in range(1, len(bounds)):
        if bounds[i] <= (bounds[i - 1] + min_w):
            bounds[i] = float(bounds[i - 1] + min_w)

    out: List[Tuple[float, float, float]] = []
    for i, c in enumerate(arr):
        l = float(bounds[i])
        r = float(bounds[i + 1])
        if not (l < float(c) < r):
            half = 0.5 * max(min_w, 0.20 * period_s)
            l = float(c - half)
            r = float(c + half)
        out.append((l, r, float(c)))
    return out


def _regularize_rep_centers_by_distance(
    centers: Sequence[float],
    start_s: float,
    end_s: float,
    target_reps: int,
    period_s: float,
) -> List[float]:
    target = max(int(target_reps), 0)
    arr = np.asarray([float(c) for c in centers if np.isfinite(c)], dtype=float)
    arr = np.unique(np.sort(arr))

    if target <= 0:
        return [float(c) for c in arr]

    vals: List[float] = [float(c) for c in arr]

    def _step(v: Sequence[float]) -> float:
        return _estimate_rep_spacing(v, start_s, end_s, target, period_s)

    def _edge_bounds(step_s: float) -> Tuple[float, float]:
        # Keep final rep centers close to the set span to avoid counting pre/post noise.
        tol = 0.08 * step_s
        return (float(start_s - tol), float(end_s + tol))

    def _remove_penalty(v: Sequence[float], idx: int) -> float:
        if not v:
            return 0.0
        step_s = _step(v)
        lo, hi = _edge_bounds(step_s)
        c = float(v[idx])

        outside = 0.0
        if c < lo:
            outside = (lo - c) / step_s
        elif c > hi:
            outside = (c - hi) / step_s

        if len(v) <= 1:
            gap_pen = 0.0
        elif idx == 0:
            g = float(v[1] - v[0])
            gap_pen = abs(g - step_s) / step_s
        elif idx == (len(v) - 1):
            g = float(v[-1] - v[-2])
            gap_pen = abs(g - step_s) / step_s
        else:
            g1 = float(v[idx] - v[idx - 1])
            g2 = float(v[idx + 1] - v[idx])
            gap_pen = 0.5 * (abs(g1 - step_s) + abs(g2 - step_s)) / step_s

        edge_pen = 0.0
        if idx == 0 and c < start_s:
            edge_pen += (start_s - c) / step_s
        if idx == (len(v) - 1) and c > end_s:
            edge_pen += (c - end_s) / step_s

        return float(2.6 * outside + 1.1 * gap_pen + 0.8 * edge_pen)

    while len(vals) > target:
        penalties = [_remove_penalty(vals, i) for i in range(len(vals))]
        drop_i = int(np.argmax(np.asarray(penalties, dtype=float)))
        vals.pop(drop_i)

    while len(vals) < target:
        step_s = _step(vals)
        if not vals:
            vals.append(float(0.5 * (start_s + end_s)))
            continue
        vals = sorted([float(x) for x in vals])

        if len(vals) == 1:
            left = float(vals[0] - step_s)
            right = float(vals[0] + step_s)
            left_score = abs((start_s + 0.5 * step_s) - left)
            right_score = abs((end_s - 0.5 * step_s) - right)
            if left_score <= right_score:
                vals.insert(0, left)
            else:
                vals.append(right)
            continue

        gaps = np.diff(np.asarray(vals, dtype=float))
        if len(gaps):
            idx = int(np.argmax(gaps))
            gmax = float(gaps[idx])
            if gmax > (1.35 * step_s):
                vals.insert(idx + 1, float(0.5 * (vals[idx] + vals[idx + 1])))
                continue

        left_room = float(vals[0] - start_s)
        right_room = float(end_s - vals[-1])
        if left_room >= right_room:
            vals.insert(0, float(vals[0] - step_s))
        else:
            vals.append(float(vals[-1] + step_s))

    vals = sorted([float(x) for x in vals])
    step_s = _step(vals)
    lo, hi = _edge_bounds(step_s)

    if len(vals) >= 2:
        if vals[0] < lo:
            vals[0] = float(max(lo, vals[1] - step_s))
        if vals[-1] > hi:
            vals[-1] = float(min(hi, vals[-2] + step_s))

    vals = [float(np.clip(v, lo, hi)) for v in vals]
    vals = sorted([float(v) for v in vals if np.isfinite(v)])

    min_sep = max(0.12 * step_s, 1e-3)

    def _dedupe_close(v: Sequence[float]) -> List[float]:
        out: List[float] = []
        for x in sorted([float(u) for u in v if np.isfinite(u)]):
            if not out or (x - out[-1]) >= min_sep:
                out.append(x)
        return out

    vals = _dedupe_close(vals)

    guard = 0
    while len(vals) < target and guard < 64:
        guard += 1
        step_s = _step(vals if vals else [0.5 * (start_s + end_s)])
        lo, hi = _edge_bounds(step_s)
        min_sep = max(0.12 * step_s, 1e-3)

        if not vals:
            vals = [float(np.clip(0.5 * (start_s + end_s), lo, hi))]
            continue

        anchors = [float(start_s)] + [float(v) for v in vals] + [float(end_s)]
        gaps = np.diff(np.asarray(anchors, dtype=float))
        if len(gaps) == 0:
            vals.append(float(np.clip(0.5 * (start_s + end_s), lo, hi)))
        else:
            idx = int(np.argmax(gaps))
            c_new = float(0.5 * (anchors[idx] + anchors[idx + 1]))
            c_new = float(np.clip(c_new, lo, hi))

            arr = np.asarray(vals, dtype=float)
            if len(arr):
                nearest = float(np.min(np.abs(arr - c_new)))
                if nearest < min_sep:
                    c_new = float(np.clip(c_new + min_sep, lo, hi))
                    nearest = float(np.min(np.abs(arr - c_new)))
                    if nearest < min_sep:
                        c_new = float(np.clip(c_new - 2.0 * min_sep, lo, hi))

            vals.append(c_new)

        vals = _dedupe_close(vals)

    while len(vals) > target:
        penalties = [_remove_penalty(vals, i) for i in range(len(vals))]
        drop_i = int(np.argmax(np.asarray(penalties, dtype=float)))
        vals.pop(drop_i)

    return sorted([float(v) for v in vals if np.isfinite(v)])


def _infer_cross_set_cadence(
    consensus_sets: Sequence[Dict[str, Any]],
    period_hint_s: float,
) -> float:
    """
    Infer a shared repetition spacing from all detected sets.

    This is used to correlate sets with one another so rep spacing is consistent
    across the full session, not only inside each set.
    """
    vals: List[float] = []
    if np.isfinite(period_hint_s) and period_hint_s > 0:
        vals.append(float(period_hint_s))

    for s in consensus_sets:
        reps = int(s.get("rep_count", 0))
        start_s = float(s.get("start_s", 0.0))
        end_s = float(s.get("end_s", 0.0))
        if reps > 1 and np.isfinite(start_s) and np.isfinite(end_s) and end_s > start_s:
            vals.append(float((end_s - start_s) / float(reps - 1)))

    arr = np.asarray([float(v) for v in vals if np.isfinite(v) and v > 1e-6], dtype=float)
    if len(arr) == 0:
        return max(float(period_hint_s) if (np.isfinite(period_hint_s) and period_hint_s > 0) else 1.0, 1e-6)

    if len(arr) >= 5:
        q_lo, q_hi = np.quantile(arr, [0.2, 0.8])
        arr_trim = arr[(arr >= q_lo) & (arr <= q_hi)]
        if len(arr_trim):
            arr = arr_trim

    step_s = float(np.median(arr))
    if np.isfinite(period_hint_s) and period_hint_s > 0:
        step_s = float(np.clip(step_s, 0.55 * period_hint_s, 1.65 * period_hint_s))
    return max(step_s, 1e-6)


def _standardize_set_rep_counts_by_cadence(
    consensus_sets: List[Dict[str, Any]],
    period_hint_s: float,
) -> None:
    """
    Align per-set rep counts to a shared cadence when a neighboring count (+/-1)
    matches the cross-set cadence better.
    """
    if not consensus_sets:
        return

    step_s = _infer_cross_set_cadence(consensus_sets, period_hint_s)
    if not np.isfinite(step_s) or step_s <= 1e-6:
        return

    for s in consensus_sets:
        start_s = float(s.get("start_s", 0.0))
        end_s = float(s.get("end_s", 0.0))
        current = int(s.get("rep_count", 0))
        if current <= 0 or not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            continue

        duration_s = float(end_s - start_s)
        expected = int(max(1, round(duration_s / step_s) + 1))

        candidates = sorted({max(1, current - 1), current, current + 1, expected})

        def _score(rep_count: int) -> float:
            if rep_count <= 1:
                spacing = duration_s
            else:
                spacing = duration_s / float(rep_count - 1)
            cadence_err = abs(spacing - step_s) / step_s
            move_penalty = 0.26 * abs(rep_count - current)
            return float(cadence_err + move_penalty)

        cur_score = _score(current)
        best = min(candidates, key=_score)
        best_score = _score(best)

        if best != current and (best_score + 0.04) < cur_score:
            s["rep_count_raw_cadence"] = int(current)
            s["rep_count"] = int(best)
            prior_method = str(s.get("rep_count_method", "consensus"))
            if "cross_set_cadence" not in prior_method:
                s["rep_count_method"] = f"{prior_method}+cross_set_cadence"
            s["proper"] = bool((float(s.get("quality_score", 0.0)) >= 0.58) and (int(best) >= 3))


def _select_center_subset_by_span(
    raw_centers: Sequence[float],
    start_s: float,
    end_s: float,
    target_reps: int,
    step_s: float,
) -> List[float]:
    """
    Choose the best contiguous subset of detected centers for a target rep count.
    """
    arr = np.asarray([float(c) for c in raw_centers if np.isfinite(c)], dtype=float)
    arr = np.unique(np.sort(arr))
    target = max(int(target_reps), 0)
    if target <= 0 or len(arr) == 0:
        return []
    if len(arr) <= target:
        return [float(c) for c in arr]

    step = max(float(step_s), 1e-6)
    best_score = float("inf")
    best_block = arr[:target]

    for i in range(0, len(arr) - target + 1):
        blk = arr[i : i + target]
        edge = abs(float(blk[0]) - float(start_s)) + abs(float(blk[-1]) - float(end_s))
        if len(blk) >= 2:
            d = np.diff(blk)
            med = float(np.median(d))
            cv = float(np.std(d) / (np.mean(d) + 1e-9))
            cadence = abs(med - step)
        else:
            cv = 0.0
            cadence = 0.0

        outside = max(0.0, float(start_s) - float(blk[0])) + max(0.0, float(blk[-1]) - float(end_s))
        score = (edge / step) + 0.8 * (cadence / step) + 0.35 * cv + 1.5 * (outside / step)
        if score < best_score:
            best_score = float(score)
            best_block = blk

    return [float(c) for c in best_block]


def _score_valley_boundary_candidate(
    bounds: np.ndarray,
    synthetic_flags: np.ndarray,
    raw_centers: np.ndarray,
    start_s: float,
    end_s: float,
    step_s: float,
) -> float:
    if len(bounds) < 2:
        return float("inf")
    d = np.diff(bounds)
    if len(d) == 0 or np.any(d <= 1e-6):
        return float("inf")

    step = max(float(step_s), 1e-6)
    edge = (abs(float(bounds[0]) - float(start_s)) + abs(float(bounds[-1]) - float(end_s))) / step
    outside = (max(0.0, float(start_s) - float(bounds[0])) + max(0.0, float(bounds[-1]) - float(end_s))) / step
    med = float(np.median(d))
    cv = float(np.std(d) / (np.mean(d) + 1e-9))
    cadence = abs(med - step) / step

    if len(raw_centers):
        support_hits = 0
        for i in range(len(bounds) - 1):
            if np.any((raw_centers > float(bounds[i])) & (raw_centers < float(bounds[i + 1]))):
                support_hits += 1
        support = float(support_hits) / float(max(1, len(bounds) - 1))
    else:
        support = 0.5

    synth_frac = float(np.mean(synthetic_flags.astype(float))) if len(synthetic_flags) else 0.0
    return float(
        edge
        + 0.90 * cadence
        + 0.45 * cv
        + 3.00 * outside
        + 1.70 * (1.0 - support)
        + 0.65 * synth_frac
    )


def _snap_boundary_grid_to_valleys(
    grid: np.ndarray,
    valleys: np.ndarray,
    step_s: float,
    snap_tol_mult: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(grid, dtype=float).copy()
    synthetic = np.ones(len(bounds), dtype=bool)
    vals = np.asarray(valleys, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.unique(np.sort(vals))

    if len(vals) == 0 or len(bounds) == 0:
        return bounds, synthetic

    snap_tol = float(max(1e-6, snap_tol_mult * max(float(step_s), 1e-6)))
    used_idx = -1
    for i, g in enumerate(bounds):
        cand_idx = np.arange(used_idx + 1, len(vals), dtype=int)
        if len(cand_idx) == 0:
            break
        local = vals[cand_idx]
        j_rel = int(np.argmin(np.abs(local - float(g))))
        j = int(cand_idx[j_rel])
        v = float(vals[j])
        if abs(v - float(g)) <= snap_tol:
            bounds[i] = v
            synthetic[i] = False
            used_idx = j

    min_gap = max(1e-3, 0.08 * max(float(step_s), 1e-6))
    for i in range(1, len(bounds)):
        if bounds[i] <= (bounds[i - 1] + min_gap):
            bounds[i] = float(bounds[i - 1] + min_gap)
            synthetic[i] = True

    return bounds, synthetic


def _build_valley_first_windows(
    start_s: float,
    end_s: float,
    target_reps: int,
    step_s: float,
    raw_centers: Sequence[float],
    set_valleys: Sequence[float],
    all_valleys: Sequence[float],
) -> List[Tuple[float, float, float, bool]]:
    target = max(int(target_reps), 0)
    if target <= 0:
        return []

    step = max(float(step_s), 1e-6)
    raw = np.asarray([float(c) for c in raw_centers if np.isfinite(c)], dtype=float)
    raw = np.unique(np.sort(raw))

    set_vals = np.asarray([float(v) for v in set_valleys if np.isfinite(v)], dtype=float)
    set_vals = np.unique(np.sort(set_vals))
    all_vals = np.asarray([float(v) for v in all_valleys if np.isfinite(v)], dtype=float)
    all_vals = np.unique(np.sort(all_vals))
    if len(all_vals) == 0:
        all_vals = set_vals

    needed = target + 1
    candidates: List[Tuple[np.ndarray, np.ndarray]] = []
    have_valley_candidate = False

    if len(set_vals) >= needed:
        for i in range(0, len(set_vals) - needed + 1):
            blk = set_vals[i : i + needed].copy()
            syn = np.zeros(needed, dtype=bool)
            candidates.append((blk, syn))
        have_valley_candidate = True

    if len(set_vals) == target and target >= 1:
        before_all = all_vals[all_vals < float(set_vals[0])]
        after_all = all_vals[all_vals > float(set_vals[-1])]

        if (
            len(before_all)
            and (float(set_vals[0]) - float(before_all[-1])) <= (1.9 * step)
            and (float(start_s) - float(before_all[-1])) <= (0.85 * step)
        ):
            b0 = float(before_all[-1])
            b0_syn = False
        else:
            b0 = float(max(float(start_s) - 0.45 * step, float(set_vals[0]) - step))
            b0_syn = True

        if (
            len(after_all)
            and (float(after_all[0]) - float(set_vals[-1])) <= (1.9 * step)
            and (float(after_all[0]) - float(end_s)) <= (0.85 * step)
        ):
            bN = float(after_all[0])
            bN_syn = False
        else:
            bN = float(min(float(end_s) + 0.45 * step, float(set_vals[-1]) + step))
            bN_syn = True

        left_bounds = np.r_[b0, set_vals]
        left_syn = np.r_[np.asarray([b0_syn], dtype=bool), np.zeros(len(set_vals), dtype=bool)]
        right_bounds = np.r_[set_vals, bN]
        right_syn = np.r_[np.zeros(len(set_vals), dtype=bool), np.asarray([bN_syn], dtype=bool)]
        added_local = False
        if float(left_bounds[0]) >= (float(start_s) - 0.20 * step):
            candidates.append((left_bounds, left_syn))
            added_local = True
        if float(right_bounds[-1]) <= (float(end_s) + 0.85 * step):
            candidates.append((right_bounds, right_syn))
            added_local = True
        if not added_local:
            candidates.append((left_bounds, left_syn))
            candidates.append((right_bounds, right_syn))
        have_valley_candidate = True

    if not have_valley_candidate:
        grid = np.linspace(float(start_s), float(end_s), needed, dtype=float)
        snap_source = all_vals if len(all_vals) else set_vals
        grid_bounds, grid_syn = _snap_boundary_grid_to_valleys(grid, snap_source, step_s=step)
        candidates.append((grid_bounds, grid_syn))

    if not candidates:
        return []

    best_score = float("inf")
    best_bounds: Optional[np.ndarray] = None
    best_syn: Optional[np.ndarray] = None
    for bounds, syn in candidates:
        s = _score_valley_boundary_candidate(
            bounds=bounds,
            synthetic_flags=syn,
            raw_centers=raw,
            start_s=start_s,
            end_s=end_s,
            step_s=step,
        )
        if s < best_score:
            best_score = float(s)
            best_bounds = bounds.copy()
            best_syn = syn.copy()

    if best_bounds is None or best_syn is None:
        return []

    out: List[Tuple[float, float, float, bool]] = []
    for i in range(target):
        left = float(best_bounds[i])
        right = float(best_bounds[i + 1])
        if right <= left:
            continue

        mid = float(0.5 * (left + right))
        inside = raw[(raw > left) & (raw < right)]
        if len(inside):
            j = int(np.argmin(np.abs(inside - mid)))
            center = float(inside[j])
            center_syn = False
        else:
            center = mid
            center_syn = True

        syn = bool(best_syn[i] or best_syn[i + 1] or center_syn)
        out.append((left, right, center, syn))

    return out


def _estimate_rep_onset_starts(rep_windows: Sequence[Dict[str, Any]]) -> np.ndarray:
    """
    Derive visualization start markers for each repetition.

    Count boundaries remain valley-anchored, but onset starts are allowed earlier
    (between consecutive rep centers) so starts are not visually delayed.
    """
    if not rep_windows:
        return np.array([], dtype=float)

    ordered = sorted(
        [
            w for w in rep_windows
            if ("left" in w and "right" in w and "center" in w)
        ],
        key=lambda w: float(w["center"]),
    )
    if not ordered:
        return np.array([], dtype=float)

    left = np.asarray([float(w["left"]) for w in ordered], dtype=float)
    center = np.asarray([float(w["center"]) for w in ordered], dtype=float)
    n = len(center)
    starts = left.copy()

    if n >= 2:
        mids = 0.5 * (center[:-1] + center[1:])
        for i in range(1, n):
            candidate = float(min(left[i], mids[i - 1]))
            lo = float(center[i - 1] + 0.06 * (center[i] - center[i - 1]))
            hi = float(center[i] - 1e-3)
            starts[i] = float(np.clip(candidate, lo, hi))

    min_gap = 1e-3
    for i in range(1, n):
        if starts[i] <= (starts[i - 1] + min_gap):
            starts[i] = float(starts[i - 1] + min_gap)
        if starts[i] >= (center[i] - min_gap):
            starts[i] = float(center[i] - min_gap)

    return starts


def _estimate_peak_aligned_starts(
    rep_windows: Sequence[Dict[str, Any]],
    peak_times: Sequence[float],
) -> np.ndarray:
    """
    Force repetition starts to align with peaks.

    For rep i>1, start i is anchored to the previous rep peak. This matches the
    desired behavior where starts are peak-locked rather than valley/onset delayed.
    """
    if not rep_windows:
        return np.array([], dtype=float)

    ordered = sorted(
        [
            w for w in rep_windows
            if ("left" in w and "right" in w and "center" in w)
        ],
        key=lambda w: float(w["center"]),
    )
    if not ordered:
        return np.array([], dtype=float)

    left = np.asarray([float(w["left"]) for w in ordered], dtype=float)
    center = np.asarray([float(w["center"]) for w in ordered], dtype=float)
    n = len(center)

    peaks = np.asarray([float(p) for p in peak_times if np.isfinite(float(p))], dtype=float)
    peaks = np.unique(np.sort(peaks))

    if n >= 2:
        step_s = float(np.median(np.diff(center)))
    else:
        step_s = float(max(0.5, ordered[0]["right"] - ordered[0]["left"]))
    step_s = max(step_s, 1e-3)

    def _nearest_peak(target: float, tol: float) -> Optional[float]:
        if len(peaks) == 0:
            return None
        idx = int(np.argmin(np.abs(peaks - float(target))))
        p = float(peaks[idx])
        if abs(p - float(target)) <= tol:
            return p
        return None

    starts = np.zeros(n, dtype=float)
    tol = max(0.55 * step_s, 0.20)

    for i in range(n):
        cand: Optional[float] = None
        if i == 0:
            prev = peaks[peaks < float(center[0])]
            if len(prev):
                p0 = float(prev[-1])
                if (float(center[0]) - p0) <= (1.4 * step_s):
                    cand = p0
            if cand is None:
                cand = _nearest_peak(float(center[0]), tol)
            if cand is None:
                cand = float(center[0])
        else:
            cand = _nearest_peak(float(center[i - 1]), tol)
            if cand is None:
                cand = float(center[i - 1])

        hi = float(center[i] - 1e-3)
        lo = float(left[i] - 1.2 * step_s)
        starts[i] = float(np.clip(float(cand), lo, hi))

    for i in range(1, n):
        min_s = float(starts[i - 1] + 1e-3)
        if starts[i] < min_s:
            starts[i] = min_s
        hi = float(center[i] - 1e-3)
        if starts[i] >= hi:
            starts[i] = hi

    return starts


def _apply_set_start_anchors_to_starts(
    starts: np.ndarray,
    centers: np.ndarray,
    rep_windows: Sequence[Dict[str, Any]],
    sets: Sequence[Dict[str, Any]],
) -> np.ndarray:
    """
    Force the first start marker of each set to the set consensus start anchor.
    """
    out = np.asarray(starts, dtype=float).copy()
    ctr = np.asarray(centers, dtype=float)
    if len(out) == 0 or len(ctr) == 0 or not sets:
        return out

    for st in sets:
        s0 = float(st.get("start_s", np.nan))
        e0 = float(st.get("end_s", np.nan))
        if not np.isfinite(s0) or not np.isfinite(e0) or e0 <= s0:
            continue

        idxs = np.where((ctr >= (s0 - 1e-6)) & (ctr <= (e0 + 1e-6)))[0]
        if len(idxs) == 0:
            for i, w in enumerate(rep_windows):
                ws = float(w.get("set_start", np.nan))
                we = float(w.get("set_end", np.nan))
                if np.isfinite(ws) and np.isfinite(we) and abs(ws - s0) <= 1e-3 and abs(we - e0) <= 1e-3:
                    idxs = np.r_[idxs, i]

        if len(idxs) == 0:
            continue

        i0 = int(np.min(idxs))
        lo = float(out[i0 - 1] + 1e-3) if i0 > 0 else float(-np.inf)
        hi = float(ctr[i0] - 1e-3)
        if hi <= lo:
            continue
        out[i0] = float(np.clip(s0, lo, hi))

    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = float(out[i - 1] + 1e-3)
        if out[i] >= ctr[i]:
            out[i] = float(ctr[i] - 1e-3)
    return out


def _standardize_windows_by_cadence(
    rep_windows: Sequence[Dict[str, Any]],
    consensus_sets: Sequence[Dict[str, Any]],
    period_s: float,
    valley_times: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Rebuild repetition windows with a shared cadence so different sets follow a
    comparable temporal template.
    """
    if not rep_windows or not consensus_sets:
        return sorted(
            [
                {
                    "left": float(w["left"]),
                    "right": float(w["right"]),
                    "center": float(w["center"]),
                    "synthetic": bool(w.get("synthetic", False)),
                    "set_start": float(w.get("set_start", np.nan)),
                    "set_end": float(w.get("set_end", np.nan)),
                }
                for w in rep_windows
                if ("left" in w and "right" in w and "center" in w)
            ],
            key=lambda x: float(x["center"]),
        )

    step_s = _infer_cross_set_cadence(consensus_sets, period_s)
    valley_src: Sequence[float] = [] if valley_times is None else valley_times
    valley_arr = np.asarray([float(v) for v in valley_src if np.isfinite(float(v))], dtype=float)
    valley_arr = np.unique(np.sort(valley_arr))
    out: List[Dict[str, Any]] = []

    for st in consensus_sets:
        start_s = float(st.get("start_s", 0.0))
        end_s = float(st.get("end_s", 0.0))
        target = max(int(st.get("rep_count", 0)), 0)
        if target <= 0 or not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            continue

        raw_centers: List[float] = []
        for w in rep_windows:
            c = float(w.get("center", np.nan))
            if not np.isfinite(c):
                continue

            w_start = float(w.get("set_start", np.nan))
            w_end = float(w.get("set_end", np.nan))
            exact_set = bool(
                np.isfinite(w_start)
                and np.isfinite(w_end)
                and abs(w_start - start_s) <= 1e-6
                and abs(w_end - end_s) <= 1e-6
            )
            loose_overlap = bool((c >= (start_s - 0.60 * step_s)) and (c <= (end_s + 0.60 * step_s)))

            if exact_set or loose_overlap:
                raw_centers.append(c)
        raw_arr = np.asarray(raw_centers, dtype=float)
        raw_arr = np.unique(np.sort(raw_arr[np.isfinite(raw_arr)]))

        seed = _select_center_subset_by_span(raw_arr, start_s=start_s, end_s=end_s, target_reps=target, step_s=step_s)
        if target > 0 and len(seed) == target:
            centers = sorted([float(c) for c in seed])
        else:
            centers = _regularize_rep_centers_by_distance(
                centers=seed if len(seed) else raw_arr,
                start_s=start_s,
                end_s=end_s,
                target_reps=target,
                period_s=step_s,
            )

        if len(raw_arr):
            snapped: List[float] = []
            for c0 in centers:
                idx = int(np.argmin(np.abs(raw_arr - float(c0))))
                r0 = float(raw_arr[idx])
                if abs(r0 - float(c0)) <= (0.33 * step_s):
                    snapped.append(r0)
                else:
                    snapped.append(float(c0))
            centers = sorted(np.unique(np.asarray(snapped, dtype=float)))
            if target > 0 and len(centers) != target:
                centers = _regularize_rep_centers_by_distance(
                    centers=centers,
                    start_s=start_s,
                    end_s=end_s,
                    target_reps=target,
                    period_s=step_s,
                )

        set_valleys = valley_arr[(valley_arr >= (start_s - 0.70 * step_s)) & (valley_arr <= (end_s + 0.70 * step_s))]

        valley_first = _build_valley_first_windows(
            start_s=start_s,
            end_s=end_s,
            target_reps=target,
            step_s=step_s,
            raw_centers=raw_arr if len(raw_arr) else centers,
            set_valleys=set_valleys,
            all_valleys=valley_arr,
        )
        if len(valley_first) == target:
            for (left, right, center, synthetic_flag) in valley_first:
                out.append(
                    {
                        "left": float(left),
                        "right": float(right),
                        "center": float(center),
                        "synthetic": bool(synthetic_flag),
                        "set_start": float(start_s),
                        "set_end": float(end_s),
                    }
                )
            continue

        if len(set_valleys):
            counted = _valley_anchored_windows_from_centers(centers, set_valleys, step_s)
        else:
            counted = _midpoint_windows_from_centers(centers, step_s)
        match_tol = max(0.20 * step_s, 0.22 * period_s)

        for (left, right, center) in counted:
            near_raw = bool(len(raw_arr) and (np.min(np.abs(raw_arr - float(center))) <= match_tol))
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

    return sorted(out, key=lambda w: float(w["center"]))


def _channel_hybrid_rep_estimate(
    channel: ChannelResult,
    start_s: float,
    end_s: float,
    pad_s: float,
) -> Optional[Dict[str, Any]]:
    """
    Estimate repetitions from peak-to-peak half-distance windows, with valley support.

    This implements the requested rule:
    - find peaks and valleys
    - use half-distance between consecutive peaks to define repetition windows
    - assign repetitions from those windows, while checking valley support
    """
    win_lo = float(start_s - pad_s)
    win_hi = float(end_s + pad_s)

    peaks = np.asarray(channel.peaks_t, dtype=float)
    valleys = np.asarray(channel.valleys_t, dtype=float)

    peak_windows, peak_windows_with_valley = _count_midpoint_windows(peaks, valleys, win_lo, win_hi)
    valley_windows, _ = _count_midpoint_windows(valleys, peaks, win_lo, win_hi)

    if peak_windows <= 0 and valley_windows <= 0:
        return None

    valley_support = (
        float(peak_windows_with_valley) / float(peak_windows)
        if peak_windows > 0
        else 0.0
    )
    pair_balance = (
        float(min(peak_windows, valley_windows)) / float(max(peak_windows, valley_windows))
        if max(peak_windows, valley_windows) > 0
        else 0.0
    )

    # Primary estimate: peak windows if valley support is adequate.
    # Fallback: use strongest supported count from peak/valley windows.
    if peak_windows > 0 and valley_support >= 0.60:
        estimate = int(peak_windows)
    else:
        estimate = int(max(peak_windows_with_valley, valley_windows))
        if estimate <= 0:
            estimate = int(max(peak_windows, valley_windows))

    confidence = float(np.clip(0.35 + 0.40 * valley_support + 0.25 * pair_balance, 0.0, 1.0))

    # Boundary compensation for half-rep cutoffs at set edges.
    # If cadence suggests exactly one more rep than window counting, and a boundary
    # is deep into a peak interval with nearby valley evidence, add +1.
    period_s = max(float(channel.period_s), 1e-6)
    duration_est = max(1, int(round((float(end_s) - float(start_s)) / period_s)) + 1)

    boundary_plus1 = 0
    if (
        duration_est == (estimate + 1)
        and len(peaks) >= 4
        and valley_support >= 0.72
        and pair_balance >= 0.62
    ):
        peaks_sorted = np.unique(np.sort(peaks))

        def _frac_between_peaks(t0: float) -> Optional[float]:
            idx = int(np.searchsorted(peaks_sorted, t0) - 1)
            j = idx + 1
            if 0 <= idx < len(peaks_sorted) and 0 <= j < len(peaks_sorted):
                left = float(peaks_sorted[idx])
                right = float(peaks_sorted[j])
                if right > left:
                    return float((t0 - left) / (right - left))
            return None

        start_frac = _frac_between_peaks(float(start_s))
        end_frac = _frac_between_peaks(float(end_s))

        start_cut = bool(start_frac is not None and start_frac >= 0.78)
        end_cut = bool(end_frac is not None and end_frac <= 0.22)

        edge_band = 0.42 * period_s
        start_edge_valley = bool(
            np.any((valleys >= (float(start_s) - edge_band)) & (valleys <= (float(start_s) + edge_band)))
        )
        end_edge_valley = bool(
            np.any((valleys >= (float(end_s) - edge_band)) & (valleys <= (float(end_s) + edge_band)))
        )

        start_ok = bool(start_cut and start_edge_valley)
        end_ok = bool(end_cut and end_edge_valley)
        if start_ok ^ end_ok:
            boundary_plus1 = 1
            estimate += 1
            confidence = float(min(1.0, confidence + 0.08))

    return {
        "estimate": max(1, int(estimate)),
        "agreement": confidence,
        "methods": {
            "peak_windows": int(peak_windows),
            "valley_windows": int(valley_windows),
            "peak_windows_with_valley": int(peak_windows_with_valley),
            "duration_est": int(duration_est),
            "boundary_plus1": int(boundary_plus1),
        },
    }


def _refine_set_rep_counts_from_windows(
    consensus_sets: List[Dict[str, Any]],
    selected_channels: Sequence[ChannelResult],
) -> None:
    """
    Refine each set rep_count with a peak-to-peak half-distance consensus.

    Per channel:
    - derive repetition windows from half-distance between consecutive peaks
    - validate with valleys
    Then aggregate channel estimates with a robust weighted median.
    """
    if not consensus_sets or not selected_channels:
        return

    channel_map = {c.name: c for c in selected_channels}

    for s in consensus_sets:
        support_names = s.get("supporting_channels") or []
        support_channels = [channel_map[n] for n in support_names if n in channel_map]
        if len(support_channels) < 2:
            support_channels = list(selected_channels)
        if not support_channels:
            continue

        periods = [float(c.period_s) for c in support_channels if np.isfinite(c.period_s) and c.period_s > 0]
        if not periods:
            continue

        period_s = float(np.median(periods))
        # Wider pad captures edge reps that start/end just outside consensus set bounds.
        pad_s = max(0.30, min(0.60 * period_s, 2.40))
        start_s = float(s["start_s"])
        end_s = float(s["end_s"])

        channel_estimates: List[int] = []
        weights: List[float] = []
        per_channel_methods: Dict[str, Dict[str, int]] = {}

        for c in support_channels:
            info = _channel_hybrid_rep_estimate(c, start_s, end_s, pad_s)
            if info is None:
                continue
            est = int(info["estimate"])
            if est <= 0:
                continue

            trust_w = float(np.clip(getattr(c, "axis_trust", 1.0), 0.30, 1.0))
            w = max(float(c.score), 0.05) * trust_w * (0.55 + 0.45 * float(info["agreement"]))
            channel_estimates.append(est)
            weights.append(w)
            per_channel_methods[c.name] = dict(info["methods"])

        if len(channel_estimates) < 2:
            continue

        arr = np.asarray(channel_estimates, dtype=float)
        w_arr = np.asarray(weights, dtype=float)

        q_lo, q_hi = np.quantile(arr, [0.15, 0.85])
        keep = (arr >= q_lo) & (arr <= q_hi)
        if int(np.sum(keep)) >= 2:
            arr = arr[keep]
            w_arr = w_arr[keep]

        robust_rep_count = _weighted_median_int([int(v) for v in arr], [float(w) for w in w_arr])
        if robust_rep_count <= 0:
            continue

        old_count = int(s["rep_count"])
        vote_weights: Dict[int, float] = {}
        for v, w in zip(arr, w_arr):
            k = int(round(float(v)))
            vote_weights[k] = float(vote_weights.get(k, 0.0) + float(w))
        robust_vote_w = float(vote_weights.get(int(robust_rep_count), 0.0))
        old_vote_w = float(vote_weights.get(int(old_count), 0.0))
        total_vote_w = float(np.sum(w_arr)) if len(w_arr) else 0.0
        robust_vote_frac = (robust_vote_w / total_vote_w) if total_vote_w > 1e-9 else 0.0

        # Do not let weak channel agreement force a +1 overcount.
        if robust_rep_count > old_count and robust_vote_w < (1.25 * max(old_vote_w, 1e-9)):
            continue
        if abs(robust_rep_count - old_count) == 1 and robust_vote_frac < 0.60:
            continue
        if abs(robust_rep_count - old_count) > 3:
            continue

        s["rep_count_raw"] = old_count
        s["rep_count"] = robust_rep_count
        s["rep_count_method"] = "peak_to_peak_half_distance"
        s["rep_count_hybrid_support"] = int(len(arr))
        s["rep_count_hybrid_median"] = float(np.median(arr))
        s["rep_count_hybrid_channel_methods"] = per_channel_methods
        s["proper"] = bool((float(s.get("quality_score", 0.0)) >= 0.58) and (robust_rep_count >= 3))


def _exercise_label(
    selected_channels: Sequence[ChannelResult],
    consensus_sets: Sequence[Dict[str, Any]],
    orientation_range: Optional[float],
) -> str:
    if not selected_channels:
        return "unknown"

    median_hz = float(np.median([c.dominant_hz for c in selected_channels]))
    median_periodicity = float(np.median([c.periodicity for c in selected_channels]))
    median_reps = float(np.median([s["rep_count"] for s in consensus_sets])) if consensus_sets else 0.0

    # Heuristic label geared to this project context.
    if (
        0.14 <= median_hz <= 0.50
        and median_periodicity >= 0.20
        and median_reps >= 3
        and (orientation_range is None or orientation_range >= 0.75)
    ):
        return "squat"

    if 0.12 <= median_hz <= 0.60 and median_periodicity >= 0.15:
        return "repetitive_strength_exercise"

    return "unknown"


def _orientation_motion_range(session_dir: Path) -> Optional[float]:
    p = session_dir / SENSOR_FILES["Orientation"]
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None

    vals: List[float] = []
    for axis in ["pitch", "roll"]:
        if axis not in df.columns:
            continue
        x = pd.to_numeric(df[axis], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 30:
            continue
        vals.append(float(np.percentile(x, 95) - np.percentile(x, 5)))
    if not vals:
        return None
    return float(np.median(vals))


def analyze_session(session_dir: Path) -> Dict[str, Any]:
    channels = _load_channels_for_session(session_dir)
    detected: List[ChannelResult] = []

    for sensor, axis, t, x in channels:
        cr = _detect_channel(t, x, sensor=sensor, axis=axis)
        if cr is not None and cr.total_reps > 0:
            detected.append(cr)

    if not detected:
        return {
            "session": session_dir.name,
            "exercise": "unknown",
            "estimated_total_reps": 0,
            "estimated_proper_sets": 0,
            "sets": [],
            "top_channels": [],
            "notes": "No sufficiently periodic channels found.",
        }

    trust_scores, trust_weights = _compute_axis_trust(detected)
    for c in detected:
        c.axis_trust = float(trust_scores.get(c.name, 1.0))

    detected = sorted(
        detected,
        key=lambda c: (float(c.score) * float(trust_weights.get(c.name, 1.0))),
        reverse=True,
    )
    selected = detected[: min(TOP_CHANNELS, len(detected))]

    # Global set consensus from top channels.
    set_candidates: List[Tuple[ChannelResult, ChannelSet]] = []
    median_period = float(np.median([c.period_s for c in selected]))
    for cr in selected:
        for s in cr.sets:
            set_candidates.append((cr, s))

    consensus_sets = _cluster_set_candidates(
        set_candidates,
        window_s=max(2.0, 1.5 * median_period),
        n_selected_channels=len(selected),
    )
    _apply_turning_point_start_consensus(consensus_sets, selected)

    if GRAPH4_ONLY_LOGIC:
        for s in consensus_sets:
            s["rep_count_method"] = "graph4_waveform_valley_consensus"
        _apply_half_period_phase_shift_rep_counts(consensus_sets, selected)
    else:
        _refine_set_rep_counts_from_windows(consensus_sets, selected)
        _standardize_set_rep_counts_by_cadence(consensus_sets, period_hint_s=median_period)

    estimated_total_reps = int(sum(s["rep_count"] for s in consensus_sets))
    estimated_proper_sets = int(sum(1 for s in consensus_sets if s["proper"]))

    # Fallback when set clustering is sparse: use robust median channel count.
    if estimated_total_reps == 0:
        ch_counts = np.array([c.total_reps for c in selected], dtype=float)
        if len(ch_counts):
            q_lo, q_hi = np.quantile(ch_counts, [0.2, 0.8])
            trimmed = ch_counts[(ch_counts >= q_lo) & (ch_counts <= q_hi)]
            if len(trimmed) == 0:
                trimmed = ch_counts
            estimated_total_reps = int(round(float(np.median(trimmed))))

    orient_range = _orientation_motion_range(session_dir)
    exercise = _exercise_label(selected, consensus_sets, orient_range)

    return {
        "session": session_dir.name,
        "exercise": exercise,
        "estimated_total_reps": estimated_total_reps,
        "estimated_proper_sets": estimated_proper_sets,
        "orientation_motion_range": orient_range,
        "sets": consensus_sets,
        "top_channels": [
            {
                "channel": c.name,
                "score": round(float(c.score), 4),
                "effective_score": round(float(c.score) * float(trust_weights.get(c.name, 1.0)), 4),
                "axis_trust": round(float(c.axis_trust), 4),
                "periodicity": round(float(c.periodicity), 4),
                "snr": round(float(c.snr), 4),
                "dominant_hz": round(float(c.dominant_hz), 4),
                "template_xcorr": round(float(c.template_xcorr), 4),
                "template_dtw": (round(float(c.template_dtw), 4) if np.isfinite(c.template_dtw) else None),
                "template_subseq": round(float(c.template_subseq), 4),
                "template_match_score": round(float(c.template_match_score), 4),
                "total_reps": int(c.total_reps),
            }
            for c in selected
        ],
    }


def _iter_sessions(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p


def _print_human_report(results: Sequence[Dict[str, Any]]) -> None:
    print("=" * 88)
    print("FINALREP Exercise / Rep / Set Analysis")
    print("=" * 88)

    for r in results:
        print(f"\nSession: {r['session']}")
        print(f"  Exercise              : {r['exercise']}")
        print(f"  Estimated total reps  : {r['estimated_total_reps']}")
        print(f"  Proper sets           : {r['estimated_proper_sets']}")
        if r.get("orientation_motion_range") is not None:
            print(f"  Orientation motion    : {r['orientation_motion_range']:.3f} (p95-p5 pitch/roll)")

        sets = r.get("sets", [])
        if sets:
            print("  Sets:")
            for i, s in enumerate(sets, start=1):
                status = "proper" if s["proper"] else "questionable"
                anchor = ""
                if s.get("start_anchor"):
                    anchor = f", start_anchor={s['start_anchor']}({int(s.get('start_anchor_support', 0))})"
                print(
                    "    "
                    f"Set {i}: reps={s['rep_count']}, "
                    f"t=[{s['start_s']:.2f}, {s['end_s']:.2f}]s, "
                    f"quality={s['quality_score']:.3f} ({status}), "
                    f"support={s['channel_support']}{anchor}"
                )
        else:
            print("  Sets: none confidently detected")

        top = r.get("top_channels", [])
        if top:
            print("  Top channels:")
            for c in top[:5]:
                print(
                    "    "
                    f"{c['channel']:<24} reps={c['total_reps']:<3d} "
                    f"hz={c['dominant_hz']:.3f} score={c['score']:.3f} "
                    f"trust={c.get('axis_trust', 1.0):.2f} "
                    f"match={c.get('template_match_score', 0.0):.2f}"
                )


def _get_selected_channels(session_dir: Path) -> List[ChannelResult]:
    channels = _load_channels_for_session(session_dir)
    detected: List[ChannelResult] = []
    for sensor, axis, t, x in channels:
        cr = _detect_channel(t, x, sensor=sensor, axis=axis)
        if cr is not None and cr.total_reps > 0:
            detected.append(cr)
    trust_scores, trust_weights = _compute_axis_trust(detected)
    for c in detected:
        c.axis_trust = float(trust_scores.get(c.name, 1.0))
    detected = sorted(
        detected,
        key=lambda c: (float(c.score) * float(trust_weights.get(c.name, 1.0))),
        reverse=True,
    )
    return detected[: min(TOP_CHANNELS, len(detected))]


def _plot_set_quality(ax, sets: Sequence[Dict[str, Any]]) -> None:
    if not sets:
        ax.text(0.5, 0.5, 'No confident sets detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.arange(1, len(sets) + 1)
    quality = [float(s['quality_score']) for s in sets]
    colors = ['#2ca02c' if s['proper'] else '#d62728' for s in sets]
    ax.bar(x, quality, color=colors, alpha=0.85)
    for i, s in enumerate(sets, start=1):
        ax.text(i, quality[i - 1] + 0.02, f"{s['rep_count']} reps", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Set {i}' for i in x])
    ax.set_ylabel('Quality score')
    ax.set_title('Set Quality')
    ax.grid(True, axis='y', alpha=0.25)


def generate_session_graphs(session_dir: Path, out_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Generate detailed diagnostic graphs for one session and save PNG files.

    Returns a dict of graph name -> file path.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.facecolor': '#1e1e2e',
        'axes.facecolor': '#1e1e2e',
        'axes.edgecolor': '#444466',
        'axes.labelcolor': '#cdd6f4',
        'xtick.color': '#cdd6f4',
        'ytick.color': '#cdd6f4',
        'text.color': '#cdd6f4',
        'grid.color': '#313244',
        'grid.linestyle': '--',
        'legend.facecolor': '#313244',
        'legend.edgecolor': '#444466',
        'figure.titlesize': 13,
    })

    session_dir = Path(session_dir)
    result = analyze_session(session_dir)
    selected = _get_selected_channels(session_dir)

    if out_dir is None:
        out_dir = session_dir / 'analysis_plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_files: Dict[str, Path] = {}

    if not selected:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No channels detected for plotting', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        p0 = out_dir / '00_no_channels.png'
        fig.savefig(p0, dpi=150)
        plt.close(fig)
        plot_files['no_channels'] = p0
        return plot_files

    sets = result.get('sets', [])

    # Build raw/filtered maps for selected channels.
    raw_channels = _load_channels_for_session(session_dir)
    raw_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sensor, axis, t, x in raw_channels:
        raw_map[f"{sensor}:{axis}"] = (np.asarray(t, dtype=float), np.asarray(x, dtype=float))

    series_map: Dict[str, Dict[str, Any]] = {}

    def _nearest_idx(arr: np.ndarray, vals: np.ndarray) -> np.ndarray:
        if len(arr) == 0 or len(vals) == 0:
            return np.array([], dtype=int)
        idx = np.searchsorted(arr, vals)
        idx = np.clip(idx, 0, len(arr) - 1)
        left = np.maximum(idx - 1, 0)
        use_left = np.abs(arr[left] - vals) < np.abs(arr[idx] - vals)
        idx[use_left] = left[use_left]
        return idx.astype(int)

    def _shade_sets(ax):
        for st in sets:
            color = '#2ca02c' if st.get('proper') else '#d62728'
            ax.axvspan(float(st['start_s']), float(st['end_s']), color=color, alpha=0.10)

    for ch in selected:
        if ch.name not in raw_map:
            continue
        t_raw, x_raw = raw_map[ch.name]
        t, x = _ensure_monotonic_time(t_raw, x_raw)
        if len(t) < 30:
            continue
        fs = 1.0 / float(np.median(np.diff(t)))
        t, x, fs = _downsample_if_needed(t, x, fs)
        x_norm = _zscore(x)
        try:
            filt = _bandpass(x_norm, fs)
        except Exception:
            continue
        v_idx = _nearest_idx(t, np.asarray(ch.valleys_t, dtype=float))
        p_idx = _nearest_idx(t, np.asarray(ch.peaks_t, dtype=float))
        series_map[ch.name] = {
            't': t,
            'raw': x_norm,
            'filt': filt,
            'fs': fs,
            'v_idx': v_idx,
            'p_idx': p_idx,
        }

    # 1) Valley timeline with set overlays
    try:
        fig, ax = plt.subplots(figsize=(13, max(4.8, 0.62 * max(1, len(selected)) + 2.0)))
        for i, ch in enumerate(selected, start=1):
            ax.scatter(ch.valleys_t, np.full_like(ch.valleys_t, i), s=34, alpha=0.90)
            ax.text((ch.valleys_t[0] if len(ch.valleys_t) else 0), i + 0.12,
                    f"{ch.name}  (score={ch.score:.2f})", fontsize=8)

        for i, st in enumerate(sets, start=1):
            color = '#2ca02c' if st.get('proper') else '#d62728'
            ax.axvspan(float(st['start_s']), float(st['end_s']), color=color, alpha=0.14)
            mid = 0.5 * (float(st['start_s']) + float(st['end_s']))
            ax.text(mid, len(selected) + 0.65,
                    f"Set {i}: {st['rep_count']} reps | q={st['quality_score']:.2f}",
                    ha='center', va='bottom', fontsize=9)

        ax.set_yticks(range(1, len(selected) + 1))
        ax.set_yticklabels([c.name for c in selected])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Top channels')
        ax.set_title(f"{result['session']} | exercise={result['exercise']} | estimated reps={result['estimated_total_reps']}")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        p1 = out_dir / '01_valley_timeline.png'
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        plot_files['valley_timeline'] = p1
    except Exception as exc:
        print(f"Warning: failed valley timeline plot: {exc}")

    # 2) Channel diagnostics (score, periodicity, reps, hz)
    try:
        names = [c.name for c in selected]
        scores = np.array([float(c.score) for c in selected])
        reps = np.array([int(c.total_reps) for c in selected])
        hz = np.array([float(c.dominant_hz) for c in selected])
        per = np.array([float(c.periodicity) for c in selected])
        snr = np.array([float(c.snr) for c in selected])
        x = np.arange(len(names))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        bars = ax1.bar(x, scores, color='#1f77b4', alpha=0.85, label='score')
        ax1.plot(x, per, marker='o', color='#9467bd', linewidth=2, label='periodicity')
        for i, b in enumerate(bars):
            ax1.text(i, b.get_height() + 0.02, f"{scores[i]:.2f}", ha='center', va='bottom', fontsize=8)
        ax1.set_ylabel('Score / periodicity')
        ax1.set_title('Top Channel Quality Diagnostics')
        ax1.grid(True, axis='y', alpha=0.25)
        ax1.legend(loc='upper right', fontsize=8)

        ax2.plot(x, reps, marker='o', color='#ff7f0e', linewidth=2, label='channel reps')
        ax2.plot(x, hz, marker='s', color='#2ca02c', linewidth=2, label='dominant Hz')
        ax2.plot(x, snr, marker='^', color='#17becf', linewidth=2, label='snr')
        ax2.set_ylabel('Reps / Hz / SNR')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=24, ha='right')
        ax2.grid(True, axis='y', alpha=0.25)
        ax2.legend(loc='upper right', fontsize=8)

        fig.tight_layout()
        p2 = out_dir / '02_channel_diagnostics.png'
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        plot_files['channel_diagnostics'] = p2
    except Exception as exc:
        print(f"Warning: failed channel diagnostics plot: {exc}")

    # 3) Set quality summary
    try:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        _plot_set_quality(ax, sets)
        fig.tight_layout()
        p3 = out_dir / '03_set_quality.png'
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        plot_files['set_quality'] = p3
    except Exception as exc:
        print(f"Warning: failed set quality plot: {exc}")

    # 4) Filtered waveforms + valleys for top channels
    try:
        n_plot = min(6, len(selected))
        fig, axes = plt.subplots(n_plot, 1, figsize=(13, max(2.8 * n_plot, 5)), sharex=True)
        if n_plot == 1:
            axes = [axes]
        for ax, ch in zip(axes, selected[:n_plot]):
            d = series_map.get(ch.name)
            if d is None:
                ax.text(0.5, 0.5, f'No series for {ch.name}', transform=ax.transAxes, ha='center', va='center')
                ax.set_axis_off()
                continue
            t = d['t']
            raw = d['raw']
            filt = d['filt']
            v_idx = d['v_idx']
            p_idx = d.get('p_idx', np.array([], dtype=int))
            ax.plot(t, raw, color='#7f7f7f', alpha=0.35, linewidth=0.8, label='raw zscore')
            ax.plot(t, filt, color='#1f77b4', alpha=0.95, linewidth=1.2, label='bandpassed')
            if len(v_idx):
                ax.scatter(t[v_idx], filt[v_idx], color='#d62728', s=22, zorder=4, label='valleys')
            if len(p_idx):
                ax.scatter(t[p_idx], filt[p_idx], color='#f9e2af', edgecolor='#fab387', linewidth=0.4,
                           s=28, marker='^', zorder=4, label='peaks')
            _shade_sets(ax)
            ax.set_ylabel(ch.name, fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.legend(loc='upper right', fontsize=7)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('Top Channel Waveforms with Valley/Peak Anchors', y=0.995)
        fig.tight_layout()
        p4 = out_dir / '04_top_channel_waveforms.png'
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        plot_files['top_channel_waveforms'] = p4
    except Exception as exc:
        print(f"Warning: failed waveform plot: {exc}")

    # 5) Orientation traces with peaks/valleys + final estimated repetition chart
    try:
        op = session_dir / SENSOR_FILES['Orientation']
        if op.exists() and selected:
            odf = pd.read_csv(op)
            tcol = _find_time_col(odf)
            if tcol is not None:
                ot = pd.to_numeric(odf[tcol], errors='coerce').to_numpy(dtype=float)
                orient_cols = [c for c in ['pitch', 'roll', 'yaw', 'azimuth'] if c in odf.columns]
                orient_cols = orient_cols[:3]
                if orient_cols:
                    best = selected[0]
                    period_s = max(float(best.period_s), 1e-6)

                    best_valleys = np.asarray(best.valleys_t, dtype=float)
                    best_peaks = np.asarray(best.peaks_t, dtype=float)
                    best_valleys = np.unique(np.sort(best_valleys[np.isfinite(best_valleys)]))
                    best_peaks = np.unique(np.sort(best_peaks[np.isfinite(best_peaks)]))

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

                    def _windows_from_peaks(peaks_arr: np.ndarray) -> List[Tuple[float, float, float]]:
                        peaks_arr = np.asarray(peaks_arr, dtype=float)
                        peaks_arr = peaks_arr[np.isfinite(peaks_arr)]
                        peaks_arr = np.unique(np.sort(peaks_arr))
                        if len(peaks_arr) == 0:
                            return []
                        if len(peaks_arr) == 1:
                            c = float(peaks_arr[0])
                            return [(c - 0.5 * period_s, c + 0.5 * period_s, c)]

                        out: List[Tuple[float, float, float]] = []
                        for i, c in enumerate(peaks_arr):
                            if i == 0:
                                left = float(c - 0.5 * (peaks_arr[1] - peaks_arr[0]))
                            else:
                                left = float(0.5 * (peaks_arr[i - 1] + peaks_arr[i]))

                            if i == (len(peaks_arr) - 1):
                                right = float(c + 0.5 * (peaks_arr[-1] - peaks_arr[-2]))
                            else:
                                right = float(0.5 * (peaks_arr[i] + peaks_arr[i + 1]))

                            out.append((left, right, float(c)))
                        return out

                    def _rep_centers_for_set(st: Dict[str, Any]) -> List[float]:
                        start_s = float(st['start_s'])
                        end_s = float(st['end_s'])
                        target = max(int(st.get('rep_count', 0)), 0)
                        if target <= 0:
                            return []

                        pad_s = max(0.30, min(0.60 * period_s, 2.40))
                        win_lo = start_s - pad_s
                        win_hi = end_s + pad_s

                        peaks_in = best_peaks[(best_peaks >= win_lo) & (best_peaks <= win_hi)]
                        valleys_in = best_valleys[(best_valleys >= win_lo) & (best_valleys <= win_hi)]

                        peak_windows = [
                            (l, r, c)
                            for (l, r, c) in _windows_from_peaks(peaks_in)
                            if max(l, win_lo) <= min(r, win_hi)
                        ]

                        peak_with_valley = [
                            (l, r, c)
                            for (l, r, c) in peak_windows
                            if bool(np.any((valleys_in >= l) & (valleys_in <= r)))
                        ]

                        valley_support = (float(len(peak_with_valley)) / float(len(peak_windows))) if peak_windows else 0.0
                        if peak_windows and valley_support >= 0.60:
                            centers = [float(c) for _, _, c in peak_windows]
                        else:
                            src = peak_with_valley if peak_with_valley else peak_windows
                            centers = [float(c) for _, _, c in src]

                        duration_est = max(1, int(round((end_s - start_s) / period_s)) + 1)
                        if (target > len(centers)) and duration_est == (len(centers) + 1) and len(peaks_in) >= 3:
                            start_frac = _frac_between_peaks(peaks_in, start_s)
                            end_frac = _frac_between_peaks(peaks_in, end_s)
                            start_cut = bool(start_frac is not None and start_frac >= 0.68)
                            end_cut = bool(end_frac is not None and end_frac <= 0.32)

                            edge_band = 0.42 * period_s
                            edge_valley = bool(
                                np.any((valleys_in >= (start_s - edge_band)) & (valleys_in <= (start_s + edge_band)))
                                or np.any((valleys_in >= (end_s - edge_band)) & (valleys_in <= (end_s + edge_band)))
                            )
                            if edge_valley and (start_cut or end_cut):
                                if start_cut and not end_cut:
                                    centers.append(float(start_s - 0.25 * period_s))
                                elif end_cut and not start_cut:
                                    centers.append(float(end_s + 0.25 * period_s))
                                else:
                                    start_excess = float(start_frac - 0.5) if start_frac is not None else 0.0
                                    end_excess = float(0.5 - end_frac) if end_frac is not None else 0.0
                                    if start_excess >= end_excess:
                                        centers.append(float(start_s - 0.25 * period_s))
                                    else:
                                        centers.append(float(end_s + 0.25 * period_s))

                        centers = _regularize_rep_centers_by_distance(
                            centers=centers,
                            start_s=start_s,
                            end_s=end_s,
                            target_reps=target,
                            period_s=period_s,
                        )
                        return [float(c) for c in centers if np.isfinite(c)]

                    rep_windows_plot_raw: List[Dict[str, Any]] = []
                    for st in sets:
                        centers_st = _rep_centers_for_set(st)
                        valleys_st = best_valleys[
                            (best_valleys >= (float(st['start_s']) - 0.80 * period_s))
                            & (best_valleys <= (float(st['end_s']) + 0.80 * period_s))
                        ]
                        for (l, r, c) in _valley_anchored_windows_from_centers(centers_st, valleys_st, period_s):
                            rep_windows_plot_raw.append(
                                {
                                    'left': float(l),
                                    'right': float(r),
                                    'center': float(c),
                                    'set_start': float(st['start_s']),
                                    'set_end': float(st['end_s']),
                                }
                            )

                    rep_windows_plot = _standardize_windows_by_cadence(
                        rep_windows=rep_windows_plot_raw,
                        consensus_sets=sets,
                        period_s=period_s,
                        valley_times=best_valleys,
                    )

                    rep_centers = [float(w['center']) for w in rep_windows_plot if np.isfinite(float(w['center']))]

                    extra_sensor_channels = [ch for ch in selected[: min(5, len(selected))] if ch.name in series_map]

                    n_orient = len(orient_cols)
                    n_extra = len(extra_sensor_channels)
                    n_rows = n_orient + n_extra + 1
                    fig, axes = plt.subplots(
                        n_rows,
                        1,
                        figsize=(13, max(2.4 * n_rows, 7.4)),
                        sharex=True,
                    )
                    if n_rows == 1:
                        axes = [axes]

                    orient_axes = axes[:n_orient]
                    sensor_axes = axes[n_orient:(n_orient + n_extra)]
                    counter_ax = axes[-1]

                    rep_centers_arr = np.asarray(rep_centers, dtype=float) if rep_centers else np.array([], dtype=float)
                    rep_bounds_left_arr = (
                        np.asarray([float(w['left']) for w in rep_windows_plot], dtype=float)
                        if rep_windows_plot else np.array([], dtype=float)
                    )
                    rep_bounds_right_arr = (
                        np.asarray([float(w['right']) for w in rep_windows_plot], dtype=float)
                        if rep_windows_plot else np.array([], dtype=float)
                    )
                    rep_starts_arr = _estimate_peak_aligned_starts(rep_windows_plot, best_peaks)
                    if len(rep_starts_arr) != len(rep_centers_arr):
                        rep_starts_arr = _estimate_rep_onset_starts(rep_windows_plot)
                    if len(rep_starts_arr) != len(rep_centers_arr):
                        rep_starts_arr = rep_bounds_left_arr
                    rep_starts_arr = _apply_set_start_anchors_to_starts(
                        rep_starts_arr,
                        rep_centers_arr,
                        rep_windows_plot,
                        sets,
                    )
                    rep_ids = np.arange(1, len(rep_centers_arr) + 1, dtype=int) if len(rep_centers_arr) else np.array([], dtype=int)

                    for i, (ax, col) in enumerate(zip(orient_axes, orient_cols)):
                        sig = pd.to_numeric(odf[col], errors='coerce').to_numpy(dtype=float)
                        mask = np.isfinite(ot) & np.isfinite(sig)
                        t_sig = ot[mask]
                        s_sig = _zscore(sig[mask])
                        ax.plot(t_sig, s_sig, color='#6a5acd', linewidth=1.0, alpha=0.92, label=col)

                        for j, vt in enumerate(best_valleys):
                            ax.axvline(float(vt), color='#d62728', linewidth=0.8, alpha=0.50,
                                       label='valleys' if (i == 0 and j == 0) else None)
                        for j, pt in enumerate(best_peaks):
                            ax.axvline(float(pt), color='#f9e2af', linewidth=0.8, alpha=0.45,
                                       label='peaks' if (i == 0 and j == 0) else None)
                        for j, rs in enumerate(rep_starts_arr):
                            ax.axvline(float(rs), color='#74c7ec', linewidth=1.25, linestyle='--', alpha=0.92,
                                       label='rep starts (consensus turn)' if (i == 0 and j == 0) else None)
                        for j, rb in enumerate(rep_bounds_left_arr):
                            ax.axvline(float(rb), color='#94e2d5', linewidth=0.9, linestyle='-', alpha=0.42,
                                       label='rep boundaries' if (i == 0 and j == 0) else None)
                        for rb in rep_bounds_right_arr:
                            ax.axvline(float(rb), color='#94e2d5', linewidth=0.9, linestyle='-', alpha=0.42)
                        for j, rt in enumerate(rep_centers_arr):
                            ax.axvline(float(rt), color='#fab387', linewidth=0.9, linestyle=':',
                                       alpha=0.45, label='rep centers' if (i == 0 and j == 0) else None)

                        v_idx_sig = _nearest_idx(t_sig, best_valleys)
                        p_idx_sig = _nearest_idx(t_sig, best_peaks)
                        s_idx_sig = _nearest_idx(t_sig, rep_starts_arr)
                        if len(v_idx_sig):
                            v_t = t_sig[v_idx_sig]
                            v_y = s_sig[v_idx_sig]
                            ax.scatter(v_t, v_y, s=22, marker='v', color='#f38ba8', edgecolor='#d62728',
                                       linewidth=0.45, zorder=4)
                            for j, (vt, vy) in enumerate(zip(v_t, v_y), start=1):
                                off = -10 if (j % 2) else -16
                                ax.annotate(f"V{j}", (float(vt), float(vy)),
                                            textcoords='offset points', xytext=(0, off),
                                            ha='center', va='top', fontsize=6, color='#f38ba8',
                                            clip_on=True)
                        if len(p_idx_sig):
                            p_t = t_sig[p_idx_sig]
                            p_y = s_sig[p_idx_sig]
                            ax.scatter(p_t, p_y, s=24, marker='^', color='#f9e2af', edgecolor='#fab387',
                                       linewidth=0.45, zorder=5)
                            for j, (pt, py) in enumerate(zip(p_t, p_y), start=1):
                                off = 10 if (j % 2) else 16
                                ax.annotate(f"P{j}", (float(pt), float(py)),
                                            textcoords='offset points', xytext=(0, off),
                                            ha='center', va='bottom', fontsize=6, color='#f9e2af',
                                            clip_on=True)
                        if len(s_idx_sig):
                            s_t = t_sig[s_idx_sig]
                            s_y = s_sig[s_idx_sig]
                            ax.scatter(s_t, s_y, s=20, marker='D', color='#74c7ec', edgecolor='#89dceb',
                                       linewidth=0.4, zorder=6)

                        _shade_sets(ax)
                        ax.set_ylabel(col)
                        ax.grid(True, alpha=0.22)
                        if i == 0 and len(rep_centers_arr):
                            ylo, yhi = ax.get_ylim()
                            y_span = max(1e-6, (yhi - ylo))
                            y_rep = yhi - 0.04 * y_span
                            y_start = yhi - 0.16 * y_span
                            for rid, rs in zip(rep_ids, rep_starts_arr):
                                ax.text(float(rs), float(y_start), f"Start R{int(rid)}", ha='center', va='top',
                                        rotation=90, fontsize=6, color='#74c7ec',
                                        bbox=dict(facecolor='#1e1e2e', edgecolor='none', alpha=0.66, pad=0.2),
                                        clip_on=True)
                            for rid, rt in zip(rep_ids, rep_centers_arr):
                                ax.text(float(rt), float(y_rep), f"R{int(rid)}", ha='center', va='top',
                                        fontsize=7, color='#fab387',
                                        bbox=dict(facecolor='#1e1e2e', edgecolor='none', alpha=0.60, pad=0.3),
                                        clip_on=True)
                        if i == 0:
                            ax.legend(loc='upper right', fontsize=8)

                    for j_ax, (ax, ch) in enumerate(zip(sensor_axes, extra_sensor_channels)):
                        d = series_map.get(ch.name)
                        if d is None:
                            ax.text(0.5, 0.5, f'No series for {ch.name}', transform=ax.transAxes,
                                    ha='center', va='center')
                            ax.set_axis_off()
                            continue

                        t_ch = d['t']
                        sig_ch = d['filt']
                        v_idx_ch = d.get('v_idx', np.array([], dtype=int))
                        p_idx_ch = d.get('p_idx', np.array([], dtype=int))

                        ax.plot(t_ch, sig_ch, color='#89b4fa', linewidth=1.15, alpha=0.95, label=ch.name)
                        if len(v_idx_ch):
                            v_t = t_ch[v_idx_ch]
                            v_y = sig_ch[v_idx_ch]
                            ax.scatter(v_t, v_y, s=16, marker='v',
                                       color='#f38ba8', edgecolor='#d62728', linewidth=0.35, zorder=4)
                            for k, (vt, vy) in enumerate(zip(v_t, v_y), start=1):
                                if (k % 2) == 1:
                                    off = -8 if (k % 4) else -12
                                    ax.annotate(f"V{k}", (float(vt), float(vy)),
                                                textcoords='offset points', xytext=(0, off),
                                                ha='center', va='top', fontsize=5, color='#f38ba8',
                                                clip_on=True)
                        if len(p_idx_ch):
                            p_t = t_ch[p_idx_ch]
                            p_y = sig_ch[p_idx_ch]
                            ax.scatter(p_t, p_y, s=18, marker='^',
                                       color='#f9e2af', edgecolor='#fab387', linewidth=0.35, zorder=5)
                            for k, (pt, py) in enumerate(zip(p_t, p_y), start=1):
                                if (k % 2) == 1:
                                    off = 8 if (k % 4) else 12
                                    ax.annotate(f"P{k}", (float(pt), float(py)),
                                                textcoords='offset points', xytext=(0, off),
                                                ha='center', va='bottom', fontsize=5, color='#f9e2af',
                                                clip_on=True)

                        s_idx_ch = _nearest_idx(t_ch, rep_starts_arr)
                        if len(s_idx_ch):
                            s_t = t_ch[s_idx_ch]
                            s_y = sig_ch[s_idx_ch]
                            ax.scatter(s_t, s_y, s=16, marker='D', color='#74c7ec', edgecolor='#89dceb',
                                       linewidth=0.35, zorder=6)

                        for j, rs in enumerate(rep_starts_arr):
                            ax.axvline(float(rs), color='#74c7ec', linewidth=1.15, linestyle='--', alpha=0.86,
                                       label='rep starts (consensus turn)' if (j_ax == 0 and j == 0) else None)
                        for j, rb in enumerate(rep_bounds_left_arr):
                            ax.axvline(float(rb), color='#94e2d5', linewidth=0.85, linestyle='-', alpha=0.36,
                                       label='rep boundaries' if (j_ax == 0 and j == 0) else None)
                        for rb in rep_bounds_right_arr:
                            ax.axvline(float(rb), color='#94e2d5', linewidth=0.85, linestyle='-', alpha=0.36)
                        for j, rt in enumerate(rep_centers_arr):
                            ax.axvline(float(rt), color='#fab387', linewidth=0.85, linestyle=':', alpha=0.42,
                                       label='rep centers' if (j_ax == 0 and j == 0) else None)

                        if len(rep_starts_arr):
                            ylo_ch, yhi_ch = ax.get_ylim()
                            y_tag = yhi_ch - 0.10 * max(1e-6, (yhi_ch - ylo_ch))
                            for rid, rs in zip(rep_ids, rep_starts_arr):
                                if (int(rid) % 2) == 1:
                                    ax.text(float(rs), float(y_tag), f"S{int(rid)}", ha='center', va='top',
                                            fontsize=5, color='#74c7ec',
                                            bbox=dict(facecolor='#1e1e2e', edgecolor='none', alpha=0.58, pad=0.18),
                                            clip_on=True)

                        _shade_sets(ax)
                        ax.set_ylabel(ch.name, fontsize=8)
                        ax.grid(True, alpha=0.20)
                        if j_ax == 0:
                            ax.legend(loc='upper right', fontsize=7)

                    if len(rep_centers_arr):
                        step_start = float(min([float(st['start_s']) for st in sets])) if sets else float(rep_centers_arr[0])
                        step_x = np.r_[step_start, rep_centers_arr]
                        step_y = np.r_[0, rep_ids]
                        counter_ax.step(step_x, step_y, where='post', linewidth=2.0,
                                        color='#f9e2af', label='estimated reps')
                        counter_ax.scatter(rep_centers_arr, rep_ids, color='#fab387', s=28,
                                           zorder=3, label='rep hits')

                        if len(rep_starts_arr):
                            counter_ax.vlines(rep_starts_arr, ymin=0.0, ymax=0.35, color='#74c7ec',
                                              linewidth=1.3, linestyles='--', alpha=0.92,
                                              label='rep starts (consensus turn)')
                        if len(rep_bounds_left_arr):
                            ymax = float(np.max(rep_ids) + 0.25) if len(rep_ids) else 1.0
                            counter_ax.vlines(rep_bounds_left_arr, ymin=0.0, ymax=ymax, color='#94e2d5',
                                              linewidth=0.85, linestyles='-', alpha=0.36,
                                              label='rep boundaries')
                        if len(rep_bounds_right_arr):
                            ymax = float(np.max(rep_ids) + 0.25) if len(rep_ids) else 1.0
                            counter_ax.vlines(rep_bounds_right_arr, ymin=0.0, ymax=ymax, color='#94e2d5',
                                              linewidth=0.85, linestyles='-', alpha=0.36)

                        for rid, rs in zip(rep_ids, rep_starts_arr):
                            counter_ax.annotate(f"S{int(rid)}", (float(rs), 0.0),
                                                textcoords='offset points', xytext=(0, 5),
                                                ha='center', va='bottom', fontsize=7, color='#74c7ec')
                        for rid, rt in zip(rep_ids, rep_centers_arr):
                            counter_ax.annotate(f"R{int(rid)}", (float(rt), float(rid)),
                                                textcoords='offset points', xytext=(0, 8),
                                                ha='center', va='bottom', fontsize=7, color='#fab387')
                    else:
                        counter_ax.text(0.5, 0.5, 'No repetition hits available',
                                        transform=counter_ax.transAxes, ha='center', va='center')

                    final_reps = int(result.get('estimated_total_reps', len(rep_centers)))
                    for st in sets:
                        counter_ax.axvspan(float(st['start_s']), float(st['end_s']), color='#313244', alpha=0.10)
                    counter_ax.set_ylabel('Rep count')
                    counter_ax.set_xlabel('Time (s)')
                    counter_ax.set_title(f'Final Estimated Repetition Chart (total={final_reps})')
                    counter_ax.grid(True, axis='y', alpha=0.25)
                    counter_ax.legend(loc='upper left', fontsize=8)

                    fig.suptitle('Orientation + Top Sensor Signals with Peaks/Valleys and Rep Markers', y=0.995)
                    fig.tight_layout()
                    p5 = out_dir / '05_orientation_turning_points.png'
                    fig.savefig(p5, dpi=150)
                    plt.close(fig)
                    plot_files['orientation_turning_points'] = p5
    except Exception as exc:
        print(f"Warning: failed orientation plot: {exc}")

    # 6) Valley-indicator correlation heatmap across top channels
    try:
        if len(selected) >= 2:
            t_min = min(float(np.min(series_map[ch.name]['t'])) for ch in selected if ch.name in series_map)
            t_max = max(float(np.max(series_map[ch.name]['t'])) for ch in selected if ch.name in series_map)
            t_grid = np.linspace(t_min, t_max, 700)
            sigma = 0.7
            labels = []
            signals = []
            for ch in selected:
                if ch.name not in series_map:
                    continue
                ind = np.zeros_like(t_grid)
                for vt in np.asarray(ch.valleys_t, dtype=float):
                    ind += np.exp(-0.5 * ((t_grid - vt) / sigma) ** 2)
                ind = _zscore(ind)
                labels.append(ch.name)
                signals.append(ind)
            if len(signals) >= 2:
                M = np.corrcoef(np.vstack(signals))
                fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels) + 4), max(6, 0.9 * len(labels) + 3)))
                im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdYlGn')
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=8)
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center', fontsize=7,
                                color='white' if abs(M[i, j]) > 0.55 else 'black')
                ax.set_title('Valley-Indicator Correlation (Top Channels)')
                plt.colorbar(im, ax=ax, fraction=0.046, label='Pearson r')
                fig.tight_layout()
                p6 = out_dir / '06_valley_indicator_correlation.png'
                fig.savefig(p6, dpi=150)
                plt.close(fig)
                plot_files['valley_indicator_correlation'] = p6
    except Exception as exc:
        print(f"Warning: failed valley-indicator heatmap: {exc}")

    # 7 & 8) Best-channel rep waveform overlay + similarity heatmap
    try:
        best = selected[0]
        d = series_map.get(best.name)
        if d is not None:
            t = d['t']
            sig = d['filt']
            valleys = np.asarray(best.valleys_t, dtype=float)
            reps = []
            for i in range(len(valleys) - 1):
                lo, hi = valleys[i], valleys[i + 1]
                m = (t >= lo) & (t <= hi)
                seg = sig[m]
                if len(seg) < 12:
                    continue
                x_old = np.linspace(0, 1, len(seg))
                x_new = np.linspace(0, 1, 120)
                rs = np.interp(x_new, x_old, seg)
                reps.append(_zscore(rs))
            if len(reps) >= 1:
                reps_arr = np.vstack(reps)
                fig, ax = plt.subplots(figsize=(10, 4.8))
                for i, r in enumerate(reps_arr, start=1):
                    ax.plot(np.linspace(0, 100, reps_arr.shape[1]), r, alpha=0.75, linewidth=1.5, label=f'Rep {i}')
                ax.set_xlabel('Rep progress (%)')
                ax.set_ylabel('Normalized amplitude')
                ax.set_title(f'Best Channel Rep Overlays: {best.name}')
                ax.grid(True, alpha=0.25)
                if len(reps_arr) <= 12:
                    ax.legend(loc='upper right', fontsize=7)
                fig.tight_layout()
                p7 = out_dir / '07_best_channel_rep_overlay.png'
                fig.savefig(p7, dpi=150)
                plt.close(fig)
                plot_files['best_channel_rep_overlay'] = p7

                if len(reps_arr) >= 2:
                    sim = np.corrcoef(reps_arr)
                    fig, ax = plt.subplots(figsize=(6.8, 5.8))
                    im = ax.imshow(sim, vmin=0.4, vmax=1.0, cmap='viridis')
                    ax.set_xticks(range(len(reps_arr)))
                    ax.set_yticks(range(len(reps_arr)))
                    ax.set_xticklabels([f'R{i+1}' for i in range(len(reps_arr))])
                    ax.set_yticklabels([f'R{i+1}' for i in range(len(reps_arr))])
                    for i in range(sim.shape[0]):
                        for j in range(sim.shape[1]):
                            ax.text(j, i, f"{sim[i, j]:.2f}", ha='center', va='center', fontsize=8,
                                    color='white' if sim[i, j] > 0.75 else 'black')
                    ax.set_title(f'Rep Similarity Heatmap: {best.name}')
                    plt.colorbar(im, ax=ax, fraction=0.046, label='Correlation')
                    fig.tight_layout()
                    p8 = out_dir / '08_best_channel_rep_similarity.png'
                    fig.savefig(p8, dpi=150)
                    plt.close(fig)
                    plot_files['best_channel_rep_similarity'] = p8
    except Exception as exc:
        print(f"Warning: failed rep overlay/similarity plots: {exc}")

    # 9) Valley interval diagnostics
    try:
        labels = []
        interval_lists = []
        for ch in selected[:6]:
            d = np.diff(np.asarray(ch.valleys_t, dtype=float))
            if len(d) == 0:
                continue
            labels.append(ch.name)
            interval_lists.append(d)
        if interval_lists:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.boxplot(interval_lists, tick_labels=labels, showmeans=True)
            ax1.set_ylabel('Valley interval (s)')
            ax1.set_title('Valley Interval Distribution by Channel')
            ax1.grid(True, axis='y', alpha=0.25)
            ax1.tick_params(axis='x', rotation=20)

            for lab, d in zip(labels, interval_lists):
                ax2.plot(range(1, len(d) + 1), d, marker='o', linewidth=1.5, label=lab)
            ax2.set_xlabel('Interval index')
            ax2.set_ylabel('Interval (s)')
            ax2.set_title('Valley Interval Sequence')
            ax2.grid(True, alpha=0.25)
            ax2.legend(loc='upper right', fontsize=7)

            fig.tight_layout()
            p9 = out_dir / '09_valley_interval_diagnostics.png'
            fig.savefig(p9, dpi=150)
            plt.close(fig)
            plot_files['valley_interval_diagnostics'] = p9
    except Exception as exc:
        print(f"Warning: failed interval diagnostics plot: {exc}")

    # 10 & 11) Axis correlation heatmaps + trust score bars (sensor-level)
    try:
        sensor_mats: Dict[str, np.ndarray] = {}
        trust_scores: Dict[str, Dict[str, float]] = {}

        for table in _load_sensor_tables_for_session(session_dir):
            sensor_name = str(table["sensor"])
            if sensor_name.lower().endswith("orientation"):
                continue
            axes = dict(table["axes"])
            if any(axn not in axes for axn in ['x', 'y', 'z']):
                continue
            t = np.asarray(table["time"], dtype=float)
            data = {}
            for axn in ['x', 'y', 'z']:
                x = np.asarray(axes[axn], dtype=float)
                tt, xx = _ensure_monotonic_time(t, x)
                if len(tt) < 40:
                    continue
                fs = 1.0 / float(np.median(np.diff(tt)))
                tt, xx, fs = _downsample_if_needed(tt, xx, fs)
                xx = _zscore(xx)
                try:
                    ff = _bandpass(xx, fs)
                except Exception:
                    continue
                data[axn] = ff
            if len(data) != 3:
                continue
            L = min(len(data['x']), len(data['y']), len(data['z']))
            X = np.vstack([data['x'][:L], data['y'][:L], data['z'][:L]])
            M = np.corrcoef(X)
            sensor_mats[sensor_name] = M
            trust_scores[sensor_name] = {
                'x': float(np.mean([abs(M[0,1]), abs(M[0,2])])),
                'y': float(np.mean([abs(M[1,0]), abs(M[1,2])])),
                'z': float(np.mean([abs(M[2,0]), abs(M[2,1])])),
            }

        if sensor_mats:
            sensors = list(sensor_mats.keys())
            fig, axes = plt.subplots(1, len(sensors), figsize=(5.2 * len(sensors), 4.6))
            if len(sensors) == 1:
                axes = [axes]
            for ax, sname in zip(axes, sensors):
                M = sensor_mats[sname]
                im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdYlGn')
                ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
                ax.set_xticklabels(['X','Y','Z']); ax.set_yticklabels(['X','Y','Z'])
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, f"{M[i,j]:.2f}", ha='center', va='center', fontsize=10,
                                color='white' if abs(M[i,j]) > 0.55 else 'black')
                ax.set_title(sname)
                plt.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle('Cross-Axis Correlation Heatmaps')
            fig.tight_layout()
            p10 = out_dir / '10_axis_correlation_heatmaps.png'
            fig.savefig(p10, dpi=150)
            plt.close(fig)
            plot_files['axis_correlation_heatmaps'] = p10

            sensors = list(trust_scores.keys())
            x = np.arange(len(sensors))
            w = 0.25
            fig, ax = plt.subplots(figsize=(max(8, 2.8 * len(sensors)), 4.6))
            for i, axn in enumerate(['x','y','z']):
                vals = [trust_scores[s][axn] for s in sensors]
                bars = ax.bar(x + i*w, vals, w, label=axn.upper(), alpha=0.85)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                            f"{v:.2f}", ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x + w)
            ax.set_xticklabels(sensors, rotation=15, ha='right')
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Mean |r| with other axes')
            ax.set_title('Axis Trust Scores by Sensor')
            ax.grid(True, axis='y', alpha=0.25)
            ax.legend()
            fig.tight_layout()
            p11 = out_dir / '11_axis_trust_scores.png'
            fig.savefig(p11, dpi=150)
            plt.close(fig)
            plot_files['axis_trust_scores'] = p11
    except Exception as exc:
        print(f"Warning: failed axis correlation/trust plots: {exc}")

    # 12 & 13) Repetition windows + repetition counter (peak-to-peak method)
    try:
        best = selected[0]
        d = series_map.get(best.name)
        if d is not None and sets:
            t = d['t']
            sig = d['filt']
            v_idx = d.get('v_idx', np.array([], dtype=int))
            p_idx = d.get('p_idx', np.array([], dtype=int))

            peaks = np.asarray(best.peaks_t, dtype=float)
            peaks = peaks[np.isfinite(peaks)]
            peaks = np.unique(np.sort(peaks))

            valleys = np.asarray(best.valleys_t, dtype=float)
            valleys = valleys[np.isfinite(valleys)]
            valleys = np.unique(np.sort(valleys))

            period_s = max(float(best.period_s), 1e-6)

            def _peak_windows(events: np.ndarray) -> List[Tuple[float, float, float]]:
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

            def _windows_for_set(st: Dict[str, Any]) -> List[Dict[str, Any]]:
                start_s = float(st['start_s'])
                end_s = float(st['end_s'])
                target_reps = int(st.get('rep_count', 0))

                pad_s = max(0.30, min(0.60 * period_s, 2.40))
                win_lo = start_s - pad_s
                win_hi = end_s + pad_s

                peak_windows = [
                    (l, r, c)
                    for (l, r, c) in _peak_windows(peaks)
                    if max(l, win_lo) <= min(r, win_hi)
                ]

                peak_with_valley = [
                    (l, r, c)
                    for (l, r, c) in peak_windows
                    if bool(np.any((valleys >= l) & (valleys <= r)))
                ]

                valley_support = (float(len(peak_with_valley)) / float(len(peak_windows))) if peak_windows else 0.0

                if peak_windows and valley_support >= 0.60:
                    candidate_centers = [float(c) for (l, r, c) in peak_windows]
                else:
                    src = peak_with_valley if peak_with_valley else peak_windows
                    candidate_centers = [float(c) for (l, r, c) in src]

                # Boundary compensation for half-rep cutoffs.
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

                centers = _regularize_rep_centers_by_distance(
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
                    counted = _valley_anchored_windows_from_centers(centers, valleys_set, period_s)
                else:
                    counted = _midpoint_windows_from_centers(centers, period_s)
                raw = np.asarray([float(c) for c in candidate_centers if np.isfinite(c)], dtype=float)
                raw = np.unique(np.sort(raw))
                step_s = _estimate_rep_spacing(centers, start_s, end_s, max(target_reps, len(centers)), period_s)
                match_tol = max(0.20 * step_s, 0.22 * period_s)

                out: List[Dict[str, Any]] = []
                for (l, r, c) in counted:
                    near_raw = bool(len(raw) and np.min(np.abs(raw - float(c))) <= match_tol)
                    out.append({
                        'left': float(l),
                        'right': float(r),
                        'center': float(c),
                        'synthetic': bool(not near_raw),
                        'set_start': start_s,
                        'set_end': end_s,
                    })
                return out

            rep_windows_raw: List[Dict[str, Any]] = []
            for st in sets:
                rep_windows_raw.extend(_windows_for_set(st))

            rep_windows = _standardize_windows_by_cadence(
                rep_windows=rep_windows_raw,
                consensus_sets=sets,
                period_s=period_s,
                valley_times=valleys,
            )

            if rep_windows:
                rep_ids = np.arange(1, len(rep_windows) + 1)

                start_bounds = np.array([w['left'] for w in rep_windows], dtype=float)
                ends = np.array([w['right'] for w in rep_windows], dtype=float)
                centers = np.array([w['center'] for w in rep_windows], dtype=float)
                starts = _estimate_peak_aligned_starts(rep_windows, peaks)
                if len(starts) != len(start_bounds):
                    starts = _estimate_rep_onset_starts(rep_windows)
                if len(starts) != len(start_bounds):
                    starts = start_bounds.copy()
                starts = _apply_set_start_anchors_to_starts(starts, centers, rep_windows, sets)
                durations = np.maximum(ends - starts, 1e-6)

                # 12) Timeline with counted rep windows on waveform
                fig, ax = plt.subplots(figsize=(13, 5.8))
                ax.plot(t, sig, color='#89b4fa', linewidth=1.3, alpha=0.95, label='best-channel waveform')
                if len(v_idx):
                    v_t = t[v_idx]
                    v_y = sig[v_idx]
                    ax.scatter(v_t, v_y, s=24, color='#d62728', label='valleys', zorder=4)
                    for j, (vt, vy) in enumerate(zip(v_t, v_y), start=1):
                        off = -10 if (j % 2) else -16
                        ax.annotate(f"V{j}", (float(vt), float(vy)),
                                    textcoords='offset points', xytext=(0, off),
                                    ha='center', va='top', fontsize=7, color='#f38ba8')
                if len(p_idx):
                    p_t = t[p_idx]
                    p_y = sig[p_idx]
                    ax.scatter(p_t, p_y, s=30, marker='^', color='#f9e2af',
                               edgecolor='#fab387', linewidth=0.5, label='peaks', zorder=5)
                    for j, (pt, py) in enumerate(zip(p_t, p_y), start=1):
                        off = 10 if (j % 2) else 16
                        ax.annotate(f"P{j}", (float(pt), float(py)),
                                    textcoords='offset points', xytext=(0, off),
                                    ha='center', va='bottom', fontsize=7, color='#f9e2af')

                y_text = float(np.nanpercentile(sig, 96)) if len(sig) else 1.0
                y_low = float(np.nanpercentile(sig, 8)) if len(sig) else -1.0
                y_span = max(1e-6, (y_text - y_low))
                y_start = y_text - 0.14 * y_span
                for rid, w, s0 in zip(rep_ids, rep_windows, starts):
                    shade = '#a6e3a1' if (rid % 2 == 0) else '#89b4fa'
                    ax.axvspan(float(w['left']), float(w['right']), color=shade, alpha=0.16)
                    ax.axvline(float(w['left']), color='#94e2d5', linewidth=0.8, alpha=0.45,
                               label='rep boundaries' if int(rid) == 1 else None)
                    ax.axvline(float(s0), color='#74c7ec', linewidth=1.35, linestyle='--', alpha=0.92,
                               label='rep starts (consensus turn)' if int(rid) == 1 else None)
                    ax.axvline(float(w['right']), color='#94e2d5', linewidth=0.8, alpha=0.45,
                               label=None)
                    suffix = '*' if w['synthetic'] else ''
                    ax.text(float(s0), y_start, f"S{int(rid)}", ha='center', va='top', fontsize=7,
                            color='#74c7ec', bbox=dict(facecolor='#1e1e2e', edgecolor='none', alpha=0.65, pad=0.2))
                    ax.text(float(w['center']), y_text, f"R{int(rid)}{suffix}", ha='center', va='bottom', fontsize=8)

                _shade_sets(ax)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Filtered amplitude (z-score)')
                ax.set_title(f'Repetition Windows and Counter (Peak-to-Peak): {best.name}')
                ax.grid(True, alpha=0.28)
                ax.legend(loc='upper right', fontsize=8)
                fig.tight_layout()
                p12 = out_dir / '12_rep_boundaries_timeline.png'
                fig.savefig(p12, dpi=150)
                plt.close(fig)
                plot_files['rep_boundaries_timeline'] = p12

                # 13) Start/end summary + cumulative repetition counter
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.2), sharex=False)

                colors = ['#89b4fa' if i % 2 == 0 else '#a6e3a1' for i in range(len(rep_ids))]
                ax1.barh(
                    rep_ids,
                    durations,
                    left=starts,
                    height=0.62,
                    color=colors,
                    edgecolor='#cdd6f4',
                    alpha=0.88,
                )
                for rid, lo, hi in zip(rep_ids, starts, ends):
                    ax1.text(float(lo), float(rid) + 0.25, f"{lo:.2f}s", fontsize=7, ha='left', va='bottom')
                    ax1.text(float(hi), float(rid) - 0.25, f"{hi:.2f}s", fontsize=7, ha='right', va='top')
                    ax1.scatter([float(lo)], [float(rid)], marker='>', s=42, color='#74c7ec',
                                zorder=4, label='rep starts (consensus turn)' if int(rid) == 1 else None)
                    ax1.annotate(f"S{int(rid)}", (float(lo), float(rid)),
                                 textcoords='offset points', xytext=(-2, 9),
                                 ha='center', va='bottom', fontsize=7, color='#74c7ec')
                ax1.set_ylabel('Repetition #')
                ax1.set_xlabel('Time (s)')
                ax1.set_title('Repetition Windows (Start -> End)')
                ax1.grid(True, axis='x', alpha=0.28)

                step_x = np.r_[starts[0], centers]
                step_y = np.r_[0, rep_ids]
                ax2.step(step_x, step_y, where='post', linewidth=2.0, color='#f9e2af', label='cumulative reps')
                ax2.scatter(centers, rep_ids, color='#fab387', s=28, zorder=3, label='rep hits')
                ax2.vlines(starts, ymin=0.0, ymax=0.35, color='#74c7ec', linewidth=1.3,
                           linestyles='--', alpha=0.92, label='rep starts (consensus turn)')
                if len(start_bounds):
                    ymax = float(np.max(rep_ids) + 0.25) if len(rep_ids) else 1.0
                    ax2.vlines(start_bounds, ymin=0.0, ymax=ymax, color='#94e2d5', linewidth=0.85,
                               linestyles='-', alpha=0.40, label='rep boundaries')
                if len(ends):
                    ymax = float(np.max(rep_ids) + 0.25) if len(rep_ids) else 1.0
                    ax2.vlines(ends, ymin=0.0, ymax=ymax, color='#94e2d5', linewidth=0.85,
                               linestyles='-', alpha=0.40)
                for rid, s0 in zip(rep_ids, starts):
                    ax2.annotate(f"S{int(rid)}", (float(s0), 0.0),
                                 textcoords='offset points', xytext=(0, 5),
                                 ha='center', va='bottom', fontsize=7, color='#74c7ec')
                for rid, c in zip(rep_ids, centers):
                    ax2.annotate(f"R{int(rid)}", (float(c), float(rid)),
                                 textcoords='offset points', xytext=(0, 8),
                                 ha='center', va='bottom', fontsize=7, color='#fab387')
                for st in sets:
                    ax2.axvline(float(st['start_s']), color='#94e2d5', linewidth=0.8, alpha=0.5)
                    ax2.axvline(float(st['end_s']), color='#94e2d5', linewidth=0.8, alpha=0.5)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Repetition Count')
                ax2.set_title('Repetition Counter Over Time')
                ax2.grid(True, axis='y', alpha=0.28)
                ax2.legend(loc='upper left', fontsize=8)

                fig.tight_layout()
                p13 = out_dir / '13_rep_start_end_summary.png'
                fig.savefig(p13, dpi=150)
                plt.close(fig)
                plot_files['rep_start_end_summary'] = p13

                # 16) Dedicated repetition-start chart
                fig, ax = plt.subplots(figsize=(13, 5.6))
                ax.plot(t, sig, color='#89b4fa', linewidth=1.35, alpha=0.96, label='best-channel waveform')

                if len(v_idx):
                    ax.scatter(t[v_idx], sig[v_idx], s=20, color='#d62728', alpha=0.70, label='valleys', zorder=4)
                if len(p_idx):
                    ax.scatter(t[p_idx], sig[p_idx], s=24, marker='^', color='#f9e2af',
                               edgecolor='#fab387', linewidth=0.4, alpha=0.82, label='peaks', zorder=5)

                y_hi = float(np.nanpercentile(sig, 96)) if len(sig) else 1.0
                y_lo = float(np.nanpercentile(sig, 6)) if len(sig) else -1.0
                y_span = max(1e-6, y_hi - y_lo)
                y_ann = y_hi - 0.03 * y_span
                band_frac = 0.18

                s_idx = _nearest_idx(t, starts)
                if len(s_idx):
                    ax.scatter(t[s_idx], sig[s_idx], s=34, marker='D', color='#74c7ec',
                               edgecolor='#89dceb', linewidth=0.45, label='rep starts (consensus turn)', zorder=6)

                for rid, s0, b0, e0 in zip(rep_ids, starts, start_bounds, ends):
                    ax.axvline(float(b0), color='#94e2d5', linewidth=0.95, linestyle='-', alpha=0.48,
                               label='rep boundaries' if int(rid) == 1 else None)
                    ax.axvline(float(s0), color='#74c7ec', linewidth=1.6, linestyle='--', alpha=0.95)
                    ax.axvline(float(e0), color='#94e2d5', linewidth=0.95, linestyle='-', alpha=0.48)
                    width = max(0.12, band_frac * max(0.12, float(e0 - s0)))
                    ax.axvspan(float(s0), float(min(e0, s0 + width)), color='#74c7ec', alpha=0.16)
                    ax.text(float(s0), float(y_ann), f"Start R{int(rid)}", rotation=90,
                            ha='center', va='top', fontsize=7, color='#74c7ec',
                            bbox=dict(facecolor='#1e1e2e', edgecolor='none', alpha=0.66, pad=0.22))

                _shade_sets(ax)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Filtered amplitude (z-score)')
                ax.set_title('Repetition Starts (S# / Start R#)')
                ax.grid(True, alpha=0.28)
                ax.legend(loc='upper right', fontsize=8)
                fig.tight_layout()
                p16 = out_dir / '16_rep_start_markers.png'
                fig.savefig(p16, dpi=150)
                plt.close(fig)
                plot_files['rep_start_markers'] = p16
    except Exception as exc:
        print(f"Warning: failed repetition boundary plots: {exc}")

    # 14) Peak timeline with set overlays
    try:
        fig, ax = plt.subplots(figsize=(13, max(4.8, 0.62 * max(1, len(selected)) + 2.0)))
        for i, ch in enumerate(selected, start=1):
            peaks = np.asarray(ch.peaks_t, dtype=float)
            peaks = peaks[np.isfinite(peaks)]
            if len(peaks):
                ax.scatter(peaks, np.full_like(peaks, i), s=34, alpha=0.90, marker='^',
                           color='#f9e2af', edgecolor='#fab387', linewidth=0.5)
                label_x = float(peaks[0])
            else:
                label_x = 0.0
            ax.text(label_x, i + 0.12, f"{ch.name}  (peaks={len(peaks)})", fontsize=8)

        for i, st in enumerate(sets, start=1):
            color = '#2ca02c' if st.get('proper') else '#d62728'
            ax.axvspan(float(st['start_s']), float(st['end_s']), color=color, alpha=0.14)
            mid = 0.5 * (float(st['start_s']) + float(st['end_s']))
            ax.text(mid, len(selected) + 0.65,
                    f"Set {i}: {st['rep_count']} reps | q={st['quality_score']:.2f}",
                    ha='center', va='bottom', fontsize=9)

        ax.set_yticks(range(1, len(selected) + 1))
        ax.set_yticklabels([c.name for c in selected])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Top channels')
        ax.set_title('Peak Timeline (All Top Channels)')
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        p14 = out_dir / '14_peak_timeline.png'
        fig.savefig(p14, dpi=150)
        plt.close(fig)
        plot_files['peak_timeline'] = p14
    except Exception as exc:
        print(f"Warning: failed peak timeline plot: {exc}")

    # 15) Best-channel peak-to-peak windows used for rep assignment
    try:
        best = selected[0]
        d = series_map.get(best.name)
        if d is not None:
            t = d['t']
            sig = d['filt']
            v_idx = d.get('v_idx', np.array([], dtype=int))
            p_idx = d.get('p_idx', np.array([], dtype=int))
            peaks = np.asarray(best.peaks_t, dtype=float)
            peaks = peaks[np.isfinite(peaks)]
            peaks = np.unique(np.sort(peaks))

            fig, ax = plt.subplots(figsize=(13, 5.8))
            ax.plot(t, sig, color='#89b4fa', linewidth=1.3, alpha=0.95, label='best-channel waveform')
            if len(v_idx):
                ax.scatter(t[v_idx], sig[v_idx], s=24, color='#d62728', label='valleys', zorder=4)
            if len(p_idx):
                ax.scatter(t[p_idx], sig[p_idx], s=30, marker='^', color='#f9e2af',
                           edgecolor='#fab387', linewidth=0.5, label='peaks', zorder=5)

            win_id = 1
            y_text = float(np.nanpercentile(sig, 96)) if len(sig) else 1.0
            if len(peaks) >= 2:
                for i, pk in enumerate(peaks):
                    if i == 0:
                        left = float(pk - 0.5 * (peaks[1] - peaks[0]))
                    else:
                        left = float(0.5 * (peaks[i - 1] + peaks[i]))

                    if i == (len(peaks) - 1):
                        right = float(pk + 0.5 * (peaks[-1] - peaks[-2]))
                    else:
                        right = float(0.5 * (peaks[i] + peaks[i + 1]))

                    overlap = any(max(left, float(st['start_s'])) <= min(right, float(st['end_s'])) for st in sets)
                    if not overlap:
                        continue

                    ax.axvspan(left, right, color='#a6e3a1', alpha=0.14)
                    ax.axvline(left, color='#94e2d5', linewidth=0.8, alpha=0.6)
                    ax.axvline(right, color='#94e2d5', linewidth=0.8, alpha=0.6)
                    ax.text(0.5 * (left + right), y_text, f"R{win_id}", ha='center', va='bottom', fontsize=8)
                    win_id += 1

            _shade_sets(ax)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Filtered amplitude (z-score)')
            ax.set_title(f'Peak-to-Peak Half-Distance Windows: {best.name}')
            ax.grid(True, alpha=0.28)
            ax.legend(loc='upper right', fontsize=8)
            fig.tight_layout()
            p15 = out_dir / '15_peak_window_assignments.png'
            fig.savefig(p15, dpi=150)
            plt.close(fig)
            plot_files['peak_window_assignments'] = p15
    except Exception as exc:
        print(f"Warning: failed peak-window assignment plot: {exc}")

    return plot_files



def run_analysis(
    session_path: Optional[str] = None,
    root_path: Optional[str] = None,
    json_out: Optional[str] = None,
    plots: bool = False,
    plots_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Notebook-friendly entrypoint.

    Set exactly one of:
      - session_path: single session directory
      - root_path: directory containing multiple session subdirectories
    """
    if (session_path is None) == (root_path is None):
        raise ValueError("Set exactly one of session_path or root_path.")

    def _looks_like_drive_path(raw: Optional[str]) -> bool:
        if not raw:
            return False
        s = raw.strip()
        return s.startswith("/content/drive/") or s.startswith("MyDrive/") or s.startswith("drive/MyDrive/")

    if _running_in_colab() and (
        _looks_like_drive_path(session_path)
        or _looks_like_drive_path(root_path)
        or _looks_like_drive_path(json_out)
        or _looks_like_drive_path(plots_dir)
    ):
        mounted = _mount_google_drive(force_remount=False)
        if not mounted:
            raise RuntimeError(
                "Google Drive could not be mounted. Run:\n"
                "from google.colab import drive; drive.mount('/content/drive')"
            )

    sessions: List[Path]
    if session_path is not None:
        session = _resolve_input_path(session_path, prefer_colab_drive=_running_in_colab())
        if not session.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session}")
        sessions = [session]
    else:
        assert root_path is not None
        root = _resolve_input_path(root_path, prefer_colab_drive=_running_in_colab())
        if not root.is_dir():
            raise FileNotFoundError(f"Root directory not found: {root}")
        sessions = list(_iter_sessions(root))
        if not sessions:
            raise FileNotFoundError(f"No session directories found in: {root}")

    results = [analyze_session(s) for s in sessions]
    _print_human_report(results)

    if plots:
        if len(sessions) == 1:
            plot_dir = _resolve_output_path(plots_dir) if plots_dir else (sessions[0] / "analysis_plots")
            generate_session_graphs(sessions[0], out_dir=plot_dir)
        else:
            root = sessions[0].parent
            base_plot_dir = _resolve_output_path(plots_dir) if plots_dir else (root / "analysis_plots")
            for sdir in sessions:
                generate_session_graphs(sdir, out_dir=(base_plot_dir / sdir.name))

    if json_out:
        out = _resolve_output_path(json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote JSON report: {out}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved IMU rep/set analysis")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--root", type=str, help="Directory containing multiple session folders")
    group.add_argument("--session", type=str, help="Single session folder path")
    group.add_argument(
        "--drive-root",
        type=str,
        help="Path inside MyDrive for a directory containing multiple sessions (Colab helper)",
    )
    group.add_argument(
        "--drive-session",
        type=str,
        help="Path inside MyDrive for a single session directory (Colab helper)",
    )
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help="Mount Google Drive first when running in Colab",
    )
    parser.add_argument(
        "--force-remount-drive",
        action="store_true",
        help="Force re-mount Google Drive in Colab",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional output JSON file")
    parser.add_argument("--plots", action="store_true", help="Generate diagnostic PNG graphs")
    parser.add_argument("--plots-dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    def _looks_like_drive_path(raw: Optional[str]) -> bool:
        if not raw:
            return False
        s = raw.strip()
        return s.startswith("/content/drive/") or s.startswith("MyDrive/") or s.startswith("drive/MyDrive/")

    use_colab_drive = bool(
        args.drive_root
        or args.drive_session
        or args.mount_drive
        or args.force_remount_drive
        or _looks_like_drive_path(args.session)
        or _looks_like_drive_path(args.root)
    )
    if use_colab_drive and _running_in_colab():
        mounted = _mount_google_drive(force_remount=args.force_remount_drive)
        if not mounted:
            raise SystemExit(
                "Google Drive is not available. In Colab, run with --mount-drive or mount manually:\n"
                "from google.colab import drive; drive.mount('/content/drive')"
            )

    session_arg = args.session
    root_arg = args.root

    if args.drive_session:
        session_arg = f"MyDrive/{args.drive_session.lstrip('/')}"
    elif args.drive_root:
        root_arg = f"MyDrive/{args.drive_root.lstrip('/')}"

    session: Optional[Path] = None
    root: Optional[Path] = None
    sessions: List[Path] = []

    if session_arg is None and root_arg is None:
        env_session = os.getenv("IMU_SESSION_DIR")
        env_root = os.getenv("IMU_ROOT_DIR")
        if env_session:
            session_arg = env_session
        elif env_root:
            root_arg = env_root
        elif _running_in_colab():
            if DEFAULT_COLAB_SESSION:
                session_arg = DEFAULT_COLAB_SESSION
            elif DEFAULT_COLAB_ROOT:
                root_arg = DEFAULT_COLAB_ROOT

    session: Optional[Path] = None
    root: Optional[Path] = None
    sessions: List[Path] = []

    if session_arg is None and root_arg is None:
        parser.error(
            "Provide one input path via --session/--root or --drive-session/--drive-root.\n"
            "You can also set IMU_SESSION_DIR or IMU_ROOT_DIR."
        )

    if session_arg:
        session = _resolve_input_path(session_arg, prefer_colab_drive=_running_in_colab())
        if not session.is_dir():
            raise SystemExit(f"Session directory not found: {session}")
        results = [analyze_session(session)]
    else:
        assert root_arg is not None
        root = _resolve_input_path(root_arg, prefer_colab_drive=_running_in_colab())
        if not root.is_dir():
            raise SystemExit(f"Root directory not found: {root}")
        sessions = list(_iter_sessions(root))
        if not sessions:
            raise SystemExit(f"No session directories found in: {root}")
        results = [analyze_session(s) for s in sessions]

    _print_human_report(results)

    if args.plots:
        if session is not None:
            plot_dir = _resolve_output_path(args.plots_dir) if args.plots_dir else (session / "analysis_plots")
            plot_files = generate_session_graphs(session, out_dir=plot_dir)
            print("\nGenerated plots:")
            for k, v in plot_files.items():
                print(f"  {k:20s} -> {v}")
        else:
            assert root is not None
            base_plot_dir = _resolve_output_path(args.plots_dir) if args.plots_dir else (root / "analysis_plots")
            print("\nGenerated plots:")
            for sdir in sessions:
                subdir = base_plot_dir / sdir.name
                plot_files = generate_session_graphs(sdir, out_dir=subdir)
                print(f"  {sdir.name}")
                for k, v in plot_files.items():
                    print(f"    {k:18s} -> {v}")

    if args.json_out:
        out = _resolve_output_path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote JSON report: {out}")


if __name__ == "__main__":
    main()
