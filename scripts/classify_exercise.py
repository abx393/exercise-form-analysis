"""
classify_exercise.py
---------------------
Classifies repetition-based exercises from multi-device accelerometer
(and optionally gyroscope) data.

Directory structure expected:
    <data_dir>/
        <exercise_name>/          e.g. situp, pushup, squat
            <recording_id>/       e.g. 1, 2, 3
                <sensor_files>    CSV files — see below

Supported file formats
----------------------
Garmin / Samsung / Bose (already-converted):
    Columns: absolute_timestamp, accel_x, accel_y, accel_z
    Gyro:    absolute_timestamp, gyro_x, gyro_y, gyro_z

Apple iPhone (Accelerometer.csv / Gyroscope.csv):
    Columns: time, seconds_elapsed, x, y, z
    Timestamp: per-row nanosecond epoch in `time`, or fixed epoch + seconds_elapsed

Apple AirPods (Headphone.csv):
    Columns: time, seconds_elapsed, accelerationX/Y/Z, rotationRateX/Y/Z, ...
    Yields two signal streams: headphones/accel and headphones/gyro

Device and sensor type are inferred from filename keywords:
    Device:
      "bose" | "headphone" | "Headphone"   → headphones
      "garmin" | "watch"                   → watch
      "samsung" | "phone" | "Accelerometer"
        | "Gyroscope"                      → phone
    Sensor:
      "gyro" | "Gyroscope" | "rotationRate"
        | "gyroscope" in name              → gyro
      otherwise                            → accel

Rep segmentation uses the primary device's accelerometer signal:
    1. Low-pass filter the vector magnitude
    2. Estimate rep period from the autocorrelation
    3. Detect valleys with that period as the minimum separation
    4. Apply the same time-boundary segments to all signals

Features are keyed as {device}_{sensor}_{feature}, e.g.:
    watch_accel_mag_mean, headphones_gyro_corr_xy

Model: Random Forest (recording-level train/test split to avoid data leakage).

Usage:
    python classify_exercise.py <data_dir> [--primary-device watch]
                                            [--lowpass-hz 5.0]
                                            [--prominence 0.5]
                                            [--test-size 0.2]
                                            [--n-estimators 200]
                                            [--save-model model.joblib]
                                            [--save-features features.csv]
                                            [--save-plot importance.png]
"""

import argparse
import csv
import datetime
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GARMIN_EPOCH_OFFSET = 631065600   # seconds between 1989-12-31 and 1970-01-01

# Maps lowercase filename substrings → canonical device label
DEVICE_KEYWORDS = {
    'headphones': ['bose', 'headphone'],   # covers "Headphone.csv" too
    'watch':      ['garmin', 'watch'],
    'phone':      ['samsung', 'phone',
                   'accelerometer', 'gyroscope'],  # Apple sensor filenames
}

# Maps lowercase filename substrings → canonical sensor label
# Checked after device inference; anything not matched defaults to 'accel'
GYRO_KEYWORDS = ['gyro', 'gyroscope', 'rotationrate']


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

def detect_and_convert_timestamps(raw_timestamps):
    """
    Convert raw timestamp values to Unix epoch seconds (float).

    Handles:
      - Nanoseconds  (~1e18): divide by 1e9
      - Milliseconds (~1e12): divide by 1e3
      - Seconds/Garmin epoch (~1e9): offset if pre-2010
    """
    sample = raw_timestamps[len(raw_timestamps) // 2]
    if sample >= 1e15:
        return [t / 1e9 for t in raw_timestamps]
    elif sample >= 1e11:
        return [t / 1e3 for t in raw_timestamps]
    else:
        converted = list(raw_timestamps)
        if converted[len(converted) // 2] < 1262304000:
            converted = [t + GARMIN_EPOCH_OFFSET for t in raw_timestamps]
        return converted


def build_timestamps_from_time_elapsed(time_col, elapsed_col):
    """
    Build per-row nanosecond timestamps from Apple-style (time, seconds_elapsed)
    columns, handling both fixed-epoch and per-row `time` values.

    When `time` is the same for every row (fixed recording-start epoch), the
    per-row offset comes from seconds_elapsed converted to nanoseconds using
    int64 arithmetic to avoid float64 precision loss at ~1e18 scale.
    When `time` already varies per row it is used directly.

    Returns a list of Unix-epoch-second floats (consistent with
    detect_and_convert_timestamps output).
    """
    time_int = [int(t) for t in time_col]
    if len(set(time_int)) == 1:
        # Fixed epoch: add seconds_elapsed as integer nanoseconds
        base_ns    = time_int[0]
        timestamps = [
            (base_ns + round(e * 1_000_000_000)) / 1e9
            for e in elapsed_col
        ]
    else:
        # Per-row nanosecond timestamps
        timestamps = [t / 1e9 for t in time_int]
    return timestamps


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

# Column name aliases → canonical names used internally
_ACCEL_ALIASES = {
    'x': 'xs', 'y': 'ys', 'z': 'zs',
    'accel_x': 'xs', 'accel_y': 'ys', 'accel_z': 'zs',
    'accelerationx': 'xs', 'accelerationy': 'ys', 'accelerationz': 'zs',
}
_GYRO_ALIASES = {
    'x': 'xs', 'y': 'ys', 'z': 'zs',
    'gyro_x': 'xs', 'gyro_y': 'ys', 'gyro_z': 'zs',
    'rotationratex': 'xs', 'rotationratey': 'ys', 'rotationratez': 'zs',
}


def _load_raw_csv(filepath):
    """
    Read a CSV and return (col_lower_map, rows) where col_lower_map maps
    lower-case column name → original column name, and rows is a list of dicts.
    """
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
    if not rows:
        return {}, []
    col_lower = {c.lower(): c for c in rows[0].keys()}
    return col_lower, rows


def _extract_timestamps(col_lower, rows):
    """
    Return a list of Unix-epoch-second floats from whichever timestamp
    convention the file uses (absolute_timestamp or time+seconds_elapsed).
    """
    if 'absolute_timestamp' in col_lower:
        raw = [float(r[col_lower['absolute_timestamp']]) for r in rows]
        return detect_and_convert_timestamps(raw)
    elif 'time' in col_lower and 'seconds_elapsed' in col_lower:
        time_col    = [float(r[col_lower['time']])            for r in rows]
        elapsed_col = [float(r[col_lower['seconds_elapsed']]) for r in rows]
        return build_timestamps_from_time_elapsed(time_col, elapsed_col)
    else:
        raise ValueError("No recognised timestamp columns "
                         f"(need absolute_timestamp or time+seconds_elapsed). "
                         f"Found: {list(col_lower.keys())}")


def _extract_axes(col_lower, rows, alias_map):
    """
    Extract x/y/z axes using alias_map. Returns (xs, ys, zs) lists or
    raises ValueError if the required columns are not present.
    """
    found = {}
    for src_lower, dst in alias_map.items():
        if src_lower in col_lower and dst not in found:
            found[dst] = [float(r[col_lower[src_lower]]) for r in rows]
    if not all(k in found for k in ('xs', 'ys', 'zs')):
        return None, None, None
    return found['xs'], found['ys'], found['zs']


def _sort_by_time(timestamps, xs, ys, zs):
    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    return (
        np.array([timestamps[i] for i in order]),
        np.array([xs[i]         for i in order]),
        np.array([ys[i]         for i in order]),
        np.array([zs[i]         for i in order]),
    )


def load_signal_csv(filepath, sensor):
    """
    Load a single-sensor CSV file (accel or gyro).

    Returns (ts, xs, ys, zs) as numpy arrays sorted by timestamp,
    or None if the required columns are not found.

    Parameters
    ----------
    filepath : path to CSV
    sensor   : 'accel' or 'gyro' — determines which alias map to use
    """
    col_lower, rows = _load_raw_csv(filepath)
    if not rows:
        return None

    try:
        timestamps = _extract_timestamps(col_lower, rows)
    except ValueError as e:
        print(f"    [skip] {filepath.name}: {e}")
        return None

    alias_map = _ACCEL_ALIASES if sensor == 'accel' else _GYRO_ALIASES
    xs, ys, zs = _extract_axes(col_lower, rows, alias_map)
    if xs is None:
        return None

    return _sort_by_time(timestamps, xs, ys, zs)


def load_headphone_csv(filepath):
    """
    Load an Apple Headphone.csv which contains both accelerometer and
    gyroscope signals in a single file.

    Returns a dict:
        {'accel': (ts, xs, ys, zs), 'gyro': (ts, xs, ys, zs)}
    Either value may be None if the columns are not found.
    """
    col_lower, rows = _load_raw_csv(filepath)
    if not rows:
        return {}

    try:
        timestamps = _extract_timestamps(col_lower, rows)
    except ValueError as e:
        print(f"    [skip headphone] {filepath.name}: {e}")
        return {}

    result = {}
    for sensor, alias_map in [('accel', _ACCEL_ALIASES),
                               ('gyro',  _GYRO_ALIASES)]:
        xs, ys, zs = _extract_axes(col_lower, rows, alias_map)
        if xs is not None:
            result[sensor] = _sort_by_time(timestamps, xs, ys, zs)

    return result


# ---------------------------------------------------------------------------
# Device / sensor type inference
# ---------------------------------------------------------------------------

def infer_device_and_sensor(filename):
    """
    Infer (device_label, sensor_label) from a filename.

    device_label : 'headphones' | 'watch' | 'phone' | None
    sensor_label : 'accel' | 'gyro'

    Apple Headphone.csv is a special case: it is flagged as
    (device='headphones', sensor='dual') so the caller knows to use
    load_headphone_csv() instead of load_signal_csv().

    Returns (None, None) if the device cannot be determined.
    """
    name_lower = filename.lower()

    # Infer device
    device = None
    for dev, keywords in DEVICE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            device = dev
            break
    if device is None:
        return None, None

    # Apple Headphone.csv contains both accel and gyro
    if 'headphone' in name_lower and device == 'headphones':
        # Distinguish Apple Headphone.csv (has accelerationX columns) from
        # Bose headphone accel/gyro files (already named accel_*/gyro_*)
        # We flag it as 'dual' and let load_recording handle the split.
        # Bose files have explicit 'accel' or 'gyro' in the filename.
        if not any(kw in name_lower for kw in ['accel', 'gyro']):
            return device, 'dual'

    # Infer sensor type
    sensor = 'gyro' if any(kw in name_lower for kw in GYRO_KEYWORDS) \
             else 'accel'

    return device, sensor


# ---------------------------------------------------------------------------
# Recording loader
# ---------------------------------------------------------------------------

def load_recording(recording_dir):
    """
    Load all recognised sensor files in a recording directory.

    Returns
    -------
    dict keyed by (device, sensor) tuple →
         {'ts': array, 'xs': array, 'ys': array, 'zs': array}

    Examples of keys:
        ('watch', 'accel'), ('headphones', 'accel'), ('headphones', 'gyro'),
        ('phone', 'accel'), ('phone', 'gyro')
    """
    signals = {}

    for fpath in sorted(Path(recording_dir).iterdir()):
        if fpath.suffix.lower() != '.csv':
            continue

        device, sensor = infer_device_and_sensor(fpath.name)

        if device is None:
            print(f"    [skip] unrecognised file: {fpath.name}")
            continue

        if sensor == 'dual':
            # Apple Headphone.csv — yields both accel and gyro
            streams = load_headphone_csv(fpath)
            for s, data in streams.items():
                if data is not None:
                    key = (device, s)
                    if key in signals:
                        print(f"    [skip] duplicate {key}: {fpath.name}")
                    else:
                        ts, xs, ys, zs = data
                        signals[key] = {'ts': ts, 'xs': xs,
                                        'ys': ys, 'zs': zs}
        else:
            key  = (device, sensor)
            if key in signals:
                print(f"    [skip] duplicate {key}: {fpath.name}")
                continue
            data = load_signal_csv(fpath, sensor)
            if data is None:
                print(f"    [skip] could not load {key} from {fpath.name}")
                continue
            ts, xs, ys, zs = data
            signals[key] = {'ts': ts, 'xs': xs, 'ys': ys, 'zs': zs}

    return signals


# ---------------------------------------------------------------------------
# Signal processing  (adapted from plot_multi_accel.py)
# ---------------------------------------------------------------------------

def lowpass_filter(signal, fs, cutoff_hz=5.0):
    nyq    = fs / 2.0
    cutoff = min(cutoff_hz, nyq * 0.9)
    b, a   = butter(2, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)


def compute_fs(times):
    dt = float(np.median(np.diff(times)))
    return 1.0 / dt if dt > 0 else 100.0


def estimate_rep_period_acf(mag_f, fs, min_period_s=0.3, max_period_s=10.0):
    """Return dominant rep period in seconds via normalised ACF."""
    x     = mag_f - mag_f.mean()
    n     = len(x)
    fft_x = np.fft.rfft(x, n=2 * n)
    acf   = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf   = acf / (acf[0] + 1e-12)

    min_lag = max(1, int(min_period_s * fs))
    max_lag = min(n - 1, int(max_period_s * fs))

    if min_lag >= max_lag:
        return max_period_s / 2.0

    peaks, props = find_peaks(acf[min_lag:max_lag], prominence=0.05)
    if len(peaks) == 0:
        return max_period_s / 2.0

    best     = peaks[np.argmax(props['prominences'])]
    period_s = (best + min_lag) / fs
    return period_s


def bandpass_filter(mag, fs, period_s,
                    lower_cycles=0.5, upper_cycles=2.5):
    """
    Adaptive bandpass filter whose cutoffs are derived from the estimated
    rep period, making the filter exercise-agnostic.

      low  = lower_cycles / period_s  Hz  (default 0.5× rep frequency)
             Removes DC drift and slow tilt changes.
      high = upper_cycles / period_s  Hz  (default 2.5× rep frequency)
             Keeps the fundamental and first two harmonics, cuts noise.

    Both cutoffs are clamped to valid Butterworth ranges before use.
    Falls back to returning mag unchanged if the range is degenerate.
    """
    nyq     = fs / 2.0
    low_hz  = max(lower_cycles / period_s, 0.01)
    high_hz = min(upper_cycles / period_s, nyq * 0.95)

    if low_hz >= high_hz:
        return mag.copy()

    b, a = butter(2, [low_hz / nyq, high_hz / nyq], btype='band')
    return filtfilt(b, a, mag)


def detect_valleys(times, xs, ys, zs, lowpass_hz=5.0,
                   min_separation_s=None, prominence_factor=0.5):
    """
    Detect valleys for rep segmentation.

    Pipeline:
      1. Low-pass filter → mag_f  (returned; used for feature extraction)
      2. ACF on mag_f   → rep period estimate
      3. Bandpass filter using ACF-derived period → mag_bp
         (used for valley detection only — cleaner baseline, less noise)

    Returns (valley_idx, mag_f, min_separation_s_used)
    """
    fs    = compute_fs(times)
    mag   = np.sqrt(xs**2 + ys**2 + zs**2)
    mag_f = lowpass_filter(mag, fs, lowpass_hz)

    if min_separation_s is None:
        period_s         = estimate_rep_period_acf(mag_f, fs)
        min_separation_s = period_s / 2.0
    else:
        period_s = min_separation_s * 2.0

    mag_bp = bandpass_filter(mag_f, fs, period_s)

    dist    = max(1, int(min_separation_s * fs))
    iqr     = float(np.percentile(mag_bp, 75) - np.percentile(mag_bp, 25))
    prom    = max(iqr * prominence_factor, 1e-6)
    vidx, _ = find_peaks(-mag_bp, distance=dist, prominence=prom)
    return vidx, mag_f, min_separation_s


def segment_reps(valley_idx, min_dur_factor=0.5):
    """Return list of (start_idx, end_idx) valley-to-valley segments."""
    if len(valley_idx) < 2:
        return []
    durs       = np.diff(valley_idx)
    median_dur = float(np.median(durs))
    min_dur    = max(1, int(median_dur * min_dur_factor))
    return [
        (int(valley_idx[i]), int(valley_idx[i + 1]))
        for i in range(len(valley_idx) - 1)
        if valley_idx[i + 1] - valley_idx[i] >= min_dur
    ]


def trim_to_window(ts, xs, ys, zs, t_start, t_end):
    mask = (ts >= t_start) & (ts <= t_end)
    return ts[mask], xs[mask], ys[mask], zs[mask]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def dominant_frequency(signal, fs):
    """Frequency (Hz) of the highest-amplitude FFT bin above 0 Hz."""
    n      = len(signal)
    if n < 4:
        return 0.0
    fft    = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    fft[0] = 0.0   # zero DC
    return float(freqs[np.argmax(fft)])


def top3_power_ratio(signal):
    """Fraction of total FFT power in the top 3 magnitude bins."""
    fft   = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    total = fft.sum()
    if total < 1e-12:
        return 0.0
    top3  = np.sort(fft)[-3:].sum()
    return float(top3 / total)


def zero_crossing_rate(signal):
    """Number of zero crossings per sample."""
    centred = signal - signal.mean()
    zc      = np.sum(np.diff(np.sign(centred)) != 0)
    return float(zc) / max(len(signal) - 1, 1)


def safe_pearson(a, b):
    """Pearson r, returns 0 if either signal has zero variance."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_device_features(mag_f_seg, xs_seg, ys_seg, zs_seg,
                              fs, rep_duration_s):
    """
    Compute all features for one rep on one device.

    Returns a flat dict of {feature_name: value}.
    """
    feats = {}

    # --- Magnitude features (orientation-invariant) ---
    mag = mag_f_seg   # already filtered
    feats['mag_mean']     = float(np.mean(mag))
    feats['mag_std']      = float(np.std(mag))
    feats['mag_min']      = float(np.min(mag))
    feats['mag_max']      = float(np.max(mag))
    feats['mag_range']    = feats['mag_max'] - feats['mag_min']
    feats['mag_rms']      = float(np.sqrt(np.mean(mag**2)))
    feats['mag_skew']     = float(_safe_skewness(mag))
    feats['mag_kurt']     = float(_safe_kurtosis(mag))
    feats['mag_dom_freq'] = dominant_frequency(mag, fs)
    feats['mag_top3_pwr'] = top3_power_ratio(mag)

    # Fraction of rep spent above the magnitude mean (phase asymmetry)
    above_mean = float(np.sum(mag > feats['mag_mean'])) / max(len(mag), 1)
    feats['mag_frac_above_mean'] = above_mean

    # --- Per-axis features ---
    for axis_name, axis_sig in [('x', xs_seg), ('y', ys_seg), ('z', zs_seg)]:
        feats[f'{axis_name}_mean'] = float(np.mean(axis_sig))
        feats[f'{axis_name}_std']  = float(np.std(axis_sig))
        feats[f'{axis_name}_zcr']  = zero_crossing_rate(axis_sig)

    # --- Cross-axis correlations ---
    feats['corr_xy'] = safe_pearson(xs_seg, ys_seg)
    feats['corr_xz'] = safe_pearson(xs_seg, zs_seg)
    feats['corr_yz'] = safe_pearson(ys_seg, zs_seg)

    # --- Rep timing ---
    feats['rep_duration_s'] = float(rep_duration_s)

    return feats


def _safe_skewness(x):
    mu, sd = np.mean(x), np.std(x)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 3))


def _safe_kurtosis(x):
    mu, sd = np.mean(x), np.std(x)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 4) - 3.0)


def features_for_recording(signals, primary_device='watch',
                            lowpass_hz=5.0, prominence_factor=0.5):
    """
    Segment reps from the primary device's accelerometer signal and extract
    features from all available (device, sensor) streams.

    Parameters
    ----------
    signals       : dict keyed by (device, sensor) tuples, as returned by
                    load_recording()
    primary_device: device label to use for segmentation (default: 'watch').
                    Always uses the 'accel' sensor of this device.
                    Falls back to the first available accel signal if the
                    chosen device is not present.

    Returns
    -------
    list of dicts, one per rep. Feature names are prefixed as
    {device}_{sensor}_{feature}, e.g. 'watch_accel_mag_mean'.
    'rep_duration_s' is added once, unprefixed, as a shared feature.
    Returns empty list if segmentation fails.
    """
    if not signals:
        return []

    # ------------------------------------------------------------------
    # Find the overlapping time window across all signals
    # ------------------------------------------------------------------
    t_start = max(d['ts'][0]  for d in signals.values()) + 2
    t_end   = min(d['ts'][-1] for d in signals.values()) - 2
    if t_start >= t_end:
        print("    [skip] no overlapping time window")
        return []

    # Trim all signals to the common window and normalise time to 0
    trimmed = {}
    for key, d in signals.items():
        ts, xs, ys, zs = trim_to_window(
            d['ts'], d['xs'], d['ys'], d['zs'], t_start, t_end)
        if len(ts) < 10:
            print(f"    [skip] {key} has too few samples after trim")
            return []
        times = ts - ts[0]
        trimmed[key] = {'times': times, 'xs': xs, 'ys': ys, 'zs': zs}

    # ------------------------------------------------------------------
    # Select the primary accel signal for segmentation
    # ------------------------------------------------------------------
    primary_key = (primary_device, 'accel')
    if primary_key not in trimmed:
        # Fall back to any available accel signal
        accel_keys  = [k for k in trimmed if k[1] == 'accel']
        if not accel_keys:
            print("    [skip] no accelerometer signal available for segmentation")
            return []
        primary_key = accel_keys[0]
        print(f"    [warn] primary {(primary_device, 'accel')} not found, "
              f"using {primary_key}")

    pd_sig = trimmed[primary_key]

    valley_idx, _, min_sep = detect_valleys(
        pd_sig['times'], pd_sig['xs'], pd_sig['ys'], pd_sig['zs'],
        lowpass_hz=lowpass_hz,
        prominence_factor=prominence_factor,
    )
    segments = segment_reps(valley_idx)

    if len(segments) < 2:
        print(f"    [skip] only {len(segments)} rep(s) detected")
        return []

    print(f"    {len(segments)} reps detected "
          f"(min_sep={min_sep:.2f}s, primary={primary_key})")

    # ------------------------------------------------------------------
    # Extract features per rep per (device, sensor)
    # ------------------------------------------------------------------

    # Build the NaN stub keys once — used when a signal has too few samples
    # in a rep window. Keys follow the {device}_{sensor}_{feat} convention.
    _feat_suffixes = [
        'mag_mean', 'mag_std', 'mag_min', 'mag_max', 'mag_range',
        'mag_rms', 'mag_skew', 'mag_kurt', 'mag_dom_freq', 'mag_top3_pwr',
        'mag_frac_above_mean',
        'x_mean', 'x_std', 'x_zcr',
        'y_mean', 'y_std', 'y_zcr',
        'z_mean', 'z_std', 'z_zcr',
        'corr_xy', 'corr_xz', 'corr_yz',
    ]

    all_rep_features = []

    for s_idx, e_idx in segments:
        t_rep_start = pd_sig['times'][s_idx]
        t_rep_end   = pd_sig['times'][min(e_idx, len(pd_sig['times']) - 1)]
        rep_dur     = float(t_rep_end - t_rep_start)
        if rep_dur <= 0:
            continue

        rep_feats = {'rep_duration_s': rep_dur}

        for (device, sensor), ddata in trimmed.items():
            prefix = f'{device}_{sensor}'
            mask   = ((ddata['times'] >= t_rep_start) &
                      (ddata['times'] <= t_rep_end))
            xs_s = ddata['xs'][mask]
            ys_s = ddata['ys'][mask]
            zs_s = ddata['zs'][mask]

            if len(xs_s) < 4:
                stub = {f'{prefix}_{k}': np.nan for k in _feat_suffixes}
                rep_feats.update(stub)
                continue

            fs_d    = compute_fs(ddata['times'][mask])
            mag_s   = np.sqrt(xs_s**2 + ys_s**2 + zs_s**2)
            mag_s_f = lowpass_filter(mag_s, fs_d, lowpass_hz) \
                      if len(mag_s) > 9 else mag_s

            dev_feats = extract_device_features(
                mag_s_f, xs_s, ys_s, zs_s, fs_d, rep_dur)

            for k, v in dev_feats.items():
                if k != 'rep_duration_s':
                    rep_feats[f'{prefix}_{k}'] = v

        all_rep_features.append(rep_feats)

    return all_rep_features


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(data_dir, primary_device='watch',
                  lowpass_hz=5.0, prominence_factor=0.5):
    """
    Walk data_dir/<exercise>/<recording_id>/ and build a flat feature table.

    Returns
    -------
    rows        : list of feature dicts
    labels      : list of exercise name strings (one per rep)
    group_ids   : list of recording ID strings (one per rep, for grouped CV)
    """

    data_dir = Path(data_dir)
    rows, labels, group_ids = [], [], []

    exercise_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir())

    for ex_dir in exercise_dirs:
        exercise = ex_dir.name
        subj_dirs = sorted(
            p for p in ex_dir.iterdir() if p.is_dir())
        
        for subj_dir in subj_dirs:
            rec_dirs = sorted(
                p for p in subj_dir.iterdir() if p.is_dir())

            if not rec_dirs:
                print(f"  [{exercise}] no recording subdirectories found, skipping")
                continue

            print(f"\n[{exercise}]")
            for rec_dir in rec_dirs:
                rec_id = f"{exercise}/{subj_dir.name}/{rec_dir.name}"
                print(f"  Recording: {rec_dir.name}")

                signals = load_recording(rec_dir)
                if len(signals) == 0:
                    print("    [skip] no recognised sensor files")
                    continue

                rep_feats = features_for_recording(
                    signals,
                    primary_device=primary_device,
                    lowpass_hz=lowpass_hz,
                    prominence_factor=prominence_factor,
                )

                for rf in rep_feats:
                    rows.append(rf)
                    labels.append(exercise)
                    group_ids.append(rec_id)

    return rows, labels, group_ids


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(rows, labels, group_ids,
                       test_size=0.2, n_estimators=200, random_state=42):
    """
    Train a Random Forest with a stratified, recording-level train/test split.

    Stratification is applied at the recording level: recordings are grouped
    by exercise class, and a proportional number of recordings from each class
    are held out for the test set. This ensures every class is represented in
    both splits regardless of how many recordings each class has.

    Returns
    -------
    clf           : fitted RandomForestClassifier
    feature_names : list of feature name strings
    imputer       : fitted SimpleImputer (for saving/reuse)
    X_test        : test feature matrix
    y_test        : true labels for test set
    y_pred        : predicted labels for test set
    """
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from collections import defaultdict

    # Build DataFrame to align columns across all reps
    df = pd.DataFrame(rows)
    df = df.ffill().fillna(0.0)
    feature_names = list(df.columns)

    X = df.values.astype(float)
    y = np.array(labels)
    g = np.array(group_ids)

    # Impute any remaining NaNs with column median
    imputer = SimpleImputer(strategy='median')
    X       = imputer.fit_transform(X)

    # ------------------------------------------------------------------
    # Stratified recording-level split
    #
    # For each class, collect the unique recording IDs, shuffle them, then
    # take ceil(test_size * n_recordings) recordings as test. This mirrors
    # what StratifiedShuffleSplit does at the sample level, but applied to
    # recordings so that whole recordings are never split across train/test.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(random_state)

    # Map each recording ID to its class (every rep in a recording has the
    # same label, so we just take the first occurrence)
    rec_to_class = {}
    for label, rec in zip(y, g):
        if rec not in rec_to_class:
            rec_to_class[rec] = label

    # Group recording IDs by class
    class_to_recs = defaultdict(list)
    for rec, cls in rec_to_class.items():
        class_to_recs[cls].append(rec)

    test_recs = set()
    for cls, recs in class_to_recs.items():
        recs_shuffled = recs.copy()
        rng.shuffle(recs_shuffled)
        n_test = max(1, round(test_size * len(recs_shuffled)))
        test_recs.update(recs_shuffled[:n_test])

    test_mask  = np.array([gi in test_recs for gi in g])
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    g_train, g_test = g[train_mask], g[test_mask]

    print(f"\nTrain: {len(X_train)} reps from "
          f"{len(set(g_train))} recordings")
    print(f"Test:  {len(X_test)} reps from "
          f"{len(set(g_test))} recordings")

    # Print per-class breakdown so stratification can be verified
    from collections import Counter
    train_counts = Counter(y_train)
    test_counts  = Counter(y_test)
    all_classes  = sorted(set(y))
    col = max(len(c) for c in all_classes) + 2
    print(f"\n  {'Class':<{col}}  {'Train':>6}  {'Test':>5}")
    print(f"  {'-'*col}  {'------':>6}  {'-----':>5}")
    for cls in all_classes:
        print(f"  {cls:<{col}}  {train_counts.get(cls, 0):>6}  "
              f"{test_counts.get(cls, 0):>5}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, feature_names, imputer, X_test, y_test, y_pred


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------

def print_report(y_test, y_pred):
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred))

    print("CONFUSION MATRIX")
    classes = sorted(set(y_test) | set(y_pred))
    cm      = confusion_matrix(y_test, y_pred, labels=classes)
    col_w   = max(len(c) for c in classes) + 2
    header  = " " * col_w + "".join(f"{c:>{col_w}}" for c in classes)
    print(header)
    for true_c, row in zip(classes, cm):
        print(f"{true_c:<{col_w}}" +
              "".join(f"{v:>{col_w}}" for v in row))
    print()


def plot_feature_importance(clf, feature_names, top_n=30, save_path=None):
    importances = clf.feature_importances_
    idx         = np.argsort(importances)[-top_n:][::-1]
    top_names   = [feature_names[i] for i in idx]
    top_vals    = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    bars = ax.barh(range(top_n), top_vals[::-1],
                   color='#3498db', edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean decrease in impurity (feature importance)", fontsize=10)
    ax.set_title(f"Top {top_n} features — Random Forest", fontsize=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.grid(True, axis='x', which='minor', linestyle=':', alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Feature importance plot saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train an exercise classifier from multi-device '
                    'accelerometer data.')
    parser.add_argument('data_dir',
                        help='Root data directory. Structure: '
                             '<data_dir>/<exercise>/<recording_id>/<csvs>')
    parser.add_argument('--primary-device', default='watch',
                        choices=['watch', 'headphones', 'phone'],
                        help='Device used for rep segmentation (default: watch)')
    parser.add_argument('--lowpass-hz', type=float, default=5.0,
                        help='Low-pass filter cutoff in Hz (default: 5.0)')
    parser.add_argument('--prominence', type=float, default=0.5,
                        help='Valley prominence factor × IQR (default: 0.5)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of recordings held out for test '
                             '(default: 0.2)')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of trees in the Random Forest '
                             '(default: 200)')
    parser.add_argument('--save-model', metavar='PATH', default=None,
                        help='Save the trained model to this path '
                             '(joblib format)')
    parser.add_argument('--save-features', metavar='PATH', default=None,
                        help='Save the full feature matrix to this CSV path')
    parser.add_argument('--save-plot', metavar='PATH', default=None,
                        help='Save the feature importance plot to this path')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    print("Building dataset ...")
    rows, labels, group_ids = build_dataset(
        args.data_dir,
        primary_device=args.primary_device,
        lowpass_hz=args.lowpass_hz,
        prominence_factor=args.prominence,
    )

    if len(rows) == 0:
        print("\nNo reps extracted. Check data directory structure and "
              "device file naming.")
        return

    print(f"\nTotal reps extracted: {len(rows)}")
    from collections import Counter
    for ex, cnt in sorted(Counter(labels).items()):
        print(f"  {ex}: {cnt} reps")

    # ------------------------------------------------------------------
    # Optionally save feature matrix
    # ------------------------------------------------------------------
    if args.save_features:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.insert(0, 'exercise', labels)
        df.insert(1, 'recording', group_ids)
        df.to_csv(args.save_features, index=False)
        print(f"\nFeature matrix saved: {args.save_features}")

    # ------------------------------------------------------------------
    # Train and evaluate
    # ------------------------------------------------------------------
    n_classes    = len(set(labels))
    n_recordings = len(set(group_ids))
    if n_classes < 2:
        print("\nNeed at least 2 exercise classes to train a classifier.")
        return
    if n_recordings < 2:
        print("\nNeed at least 2 recordings to perform a train/test split.")
        return

    clf, feature_names, imputer, X_test, y_test, y_pred = train_and_evaluate(
        rows, labels, group_ids,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
    )

    print_report(y_test, y_pred)

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    if args.save_model:
        joblib.dump({'model': clf, 'imputer': imputer,
                     'feature_names': feature_names}, args.save_model)
        print(f"Model saved: {args.save_model}")

    # ------------------------------------------------------------------
    # Feature importance plot
    # ------------------------------------------------------------------
    plot_feature_importance(
        clf, feature_names,
        top_n=min(30, len(feature_names)),
        save_path=args.save_plot,
    )


if __name__ == '__main__':
    main()
