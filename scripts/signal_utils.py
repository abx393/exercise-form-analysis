"""
signal_utils.py
---------------
Shared signal processing utilities used by:
    plot_multi_accel.py   – multi-device plotting and DTW form scoring
    classify_exercise.py  – exercise classification
    autoencoder_form.py   – autoencoder-based form deviation detection

Public API
----------
Constants:
    GARMIN_EPOCH_OFFSET

Timestamp handling:
    detect_and_convert_timestamps(raw_timestamps)
    build_timestamps_from_time_elapsed(time_col, elapsed_col)

CSV loading:
    load_signal_csv(filepath, sensor)       -> (ts, xs, ys, zs) | None
    load_headphone_csv(filepath)            -> {'accel': ..., 'gyro': ...}
    load_recording(recording_dir)           -> {(device, sensor): {...}}
    infer_device_and_sensor(filename)       -> (device, sensor)

Signal processing:
    compute_fs(times)
    lowpass_filter(signal, fs, cutoff_hz)
    bandpass_filter(mag, fs, period_s, ...)
    estimate_rep_period_acf(mag_f, fs, ...)

Peak / valley detection:
    detect_peaks_valleys(times, xs, ys, zs, ...)   # full version (plot)
    detect_valleys(times, xs, ys, zs, ...)         # simplified (classify/ae)

Segmentation:
    segment_reps(valley_idx, ...)

Window utilities:
    compute_sync_window(signals, trim_margin_s)
    trim_to_window(ts, xs, ys, zs, t_start, t_end)
"""

import csv
import datetime
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GARMIN_EPOCH_OFFSET = 631065600   # seconds between 1989-12-31 and 1970-01-01

DEVICE_KEYWORDS = {
    'headphones': ['bose', 'headphone'],
    'watch':      ['garmin', 'watch'],
    'phone':      ['samsung', 'phone', 'accelerometer', 'gyroscope'],
}

GYRO_KEYWORDS = ['gyro', 'gyroscope', 'rotationrate']


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

def detect_and_convert_timestamps(raw_timestamps):
    """
    Detect timestamp units/epoch and return as Unix epoch seconds (float).

    Detection order:
      1. ~1e18 → nanoseconds  (divide by 1e9)
      2. ~1e12 → milliseconds (divide by 1e3)
      3. ~1e9  → seconds; if resulting date is before 2010 assume Garmin epoch
    """
    sample = raw_timestamps[len(raw_timestamps) // 2]

    if sample >= 1e15:
        converted = [t / 1e9 for t in raw_timestamps]
        unit = "nanoseconds"
    elif sample >= 1e11:
        converted = [t / 1e3 for t in raw_timestamps]
        unit = "milliseconds"
    else:
        converted = list(raw_timestamps)
        unit = "seconds (Unix)"
        if converted[len(converted) // 2] < 1262304000:   # 2010-01-01
            converted = [t + GARMIN_EPOCH_OFFSET for t in raw_timestamps]
            unit = "seconds (Garmin epoch -> Unix)"

    mid = converted[len(converted) // 2]
    dt  = datetime.datetime.fromtimestamp(mid, tz=datetime.timezone.utc)
    print(f"    Unit detected: {unit}")
    print(f"    Midpoint time: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    return converted


def build_timestamps_from_time_elapsed(time_col, elapsed_col):
    """
    Build per-row Unix-epoch-second timestamps from Apple-style
    (time, seconds_elapsed) columns.

    `time` may be either:
      - A fixed recording-start epoch (same nanosecond value for every row).
        In this case seconds_elapsed is added as integer nanoseconds to
        avoid float64 precision loss at ~1e18 scale.
      - Already a unique per-row nanosecond timestamp, used directly.

    Returns a list of Unix-epoch-second floats.
    """
    time_int = [int(t) for t in time_col]
    if len(set(time_int)) == 1:
        base_ns    = time_int[0]
        timestamps = [
            (base_ns + round(e * 1_000_000_000)) / 1e9
            for e in elapsed_col
        ]
    else:
        timestamps = [t / 1e9 for t in time_int]
    return timestamps


# ---------------------------------------------------------------------------
# CSV loading internals
# ---------------------------------------------------------------------------

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
    """Return (col_lower_map, rows). col_lower maps lower-cased → original name."""
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
    if not rows:
        return {}, []
    col_lower = {c.lower(): c for c in rows[0].keys()}
    return col_lower, rows


def _extract_timestamps(col_lower, rows):
    """Return list of Unix-epoch-second floats from whichever convention is used."""
    if 'absolute_timestamp' in col_lower:
        raw = [float(r[col_lower['absolute_timestamp']]) for r in rows]
        return detect_and_convert_timestamps(raw)
    elif 'time' in col_lower and 'seconds_elapsed' in col_lower:
        time_col    = [float(r[col_lower['time']])            for r in rows]
        elapsed_col = [float(r[col_lower['seconds_elapsed']]) for r in rows]
        return build_timestamps_from_time_elapsed(time_col, elapsed_col)
    else:
        raise ValueError(
            "No recognised timestamp columns "
            f"(need absolute_timestamp or time+seconds_elapsed). "
            f"Found: {list(col_lower.keys())}")


def _extract_axes(col_lower, rows, alias_map):
    """
    Extract x/y/z axes using alias_map.
    Returns (xs, ys, zs) lists, or (None, None, None) if columns not found.
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


# ---------------------------------------------------------------------------
# Public CSV loaders
# ---------------------------------------------------------------------------

def load_signal_csv(filepath, sensor='accel'):
    """
    Load a single-sensor CSV file (accel or gyro).

    Handles all supported column naming conventions and timestamp formats.
    Returns (ts, xs, ys, zs) as numpy arrays sorted by timestamp,
    or None if the required columns are not found.

    Parameters
    ----------
    filepath : path-like
    sensor   : 'accel' or 'gyro'
    """
    col_lower, rows = _load_raw_csv(filepath)
    if not rows:
        return None

    try:
        timestamps = _extract_timestamps(col_lower, rows)
    except ValueError as e:
        print(f"    [skip] {Path(filepath).name}: {e}")
        return None

    alias_map  = _ACCEL_ALIASES if sensor == 'accel' else _GYRO_ALIASES
    xs, ys, zs = _extract_axes(col_lower, rows, alias_map)
    if xs is None:
        return None

    return _sort_by_time(timestamps, xs, ys, zs)


def load_headphone_csv(filepath):
    """
    Load an Apple Headphone.csv which contains both accelerometer and
    gyroscope signals in a single file.

    Returns dict: {'accel': (ts, xs, ys, zs), 'gyro': (ts, xs, ys, zs)}
    Either value may be absent if columns are not found.
    """
    col_lower, rows = _load_raw_csv(filepath)
    if not rows:
        return {}

    try:
        timestamps = _extract_timestamps(col_lower, rows)
    except ValueError as e:
        print(f"    [skip headphone] {Path(filepath).name}: {e}")
        return {}

    result = {}
    for sensor, alias_map in [('accel', _ACCEL_ALIASES),
                               ('gyro',  _GYRO_ALIASES)]:
        xs, ys, zs = _extract_axes(col_lower, rows, alias_map)
        if xs is not None:
            result[sensor] = _sort_by_time(timestamps, xs, ys, zs)
    return result


def infer_device_and_sensor(filename):
    """
    Infer (device_label, sensor_label) from a filename.

    device_label : 'headphones' | 'watch' | 'phone' | None
    sensor_label : 'accel' | 'gyro' | 'dual' | None

    'dual' is returned for Apple Headphone.csv (contains both signals).
    (None, None) is returned when the device cannot be determined.
    """
    name_lower = filename.lower()

    device = None
    for dev, keywords in DEVICE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            device = dev
            break
    if device is None:
        return None, None

    # Apple Headphone.csv: contains both accel and gyro, no 'accel'/'gyro'
    # keyword in filename; distinguished from Bose files which do have one.
    if 'headphone' in name_lower and device == 'headphones':
        if not any(kw in name_lower for kw in ['accel', 'gyro']):
            return device, 'dual'

    sensor = 'gyro' if any(kw in name_lower for kw in GYRO_KEYWORDS) \
             else 'accel'
    return device, sensor


def load_recording(recording_dir):
    """
    Load all recognised sensor files in a recording directory.

    Returns
    -------
    dict keyed by (device, sensor) tuples:
        ('watch', 'accel'), ('headphones', 'gyro'), ('phone', 'accel'), ...
    Each value: {'ts': array, 'xs': array, 'ys': array, 'zs': array}
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
            streams = load_headphone_csv(fpath)
            for s, data in streams.items():
                key = (device, s)
                if key in signals:
                    print(f"    [skip] duplicate {key}: {fpath.name}")
                else:
                    ts, xs, ys, zs = data
                    signals[key] = {'ts': ts, 'xs': xs, 'ys': ys, 'zs': zs}
        else:
            key = (device, sensor)
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
# External rep boundary loading
# ---------------------------------------------------------------------------

def load_rep_boundaries(csv_path):
    """
    Load pre-computed rep boundaries from a CSV file produced by an external
    segmentation script.

    Expected columns (others are ignored):
        relative_path  – recording path, e.g. "exercise\\person\\session"
                         (backslash or forward slash separators both accepted)
        rep_index      – 1-based integer rep number within the recording
        start_s        – rep start time in seconds from the trimmed window start
        end_s          – rep end time in seconds from the trimmed window start

    The relative_path is normalised to a (exercise, person, session) tuple
    by splitting on both backslash and forward slash and taking the last
    three components. This makes matching robust to OS path differences.

    Returns
    -------
    dict keyed by (exercise, person, session) tuple →
         sorted list of (start_s, end_s) float pairs, one per rep in order.
    """
    import csv as _csv
    from collections import defaultdict

    boundaries = defaultdict(list)

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            # Normalise path: split on both separators, take last 3 parts
            raw_path = row['relative_path'].replace('\\', '/')
            parts    = [p for p in raw_path.split('/') if p]
            if len(parts) < 3:
                continue
            key = tuple(parts[-3:])   # (exercise, person, session)

            try:
                rep_idx = int(row['rep_index'])
                start_s = float(row['start_s'])
                end_s   = float(row['end_s'])
            except (ValueError, KeyError):
                continue

            boundaries[key].append((rep_idx, start_s, end_s))

    # Sort each recording's reps by rep_index and strip the index
    result = {}
    for key, reps in boundaries.items():
        reps.sort(key=lambda x: x[0])
        result[key] = [(s, e) for _, s, e in reps]

    return result


def load_eval_sessions(csv_path):
    """
    Load the set of recordings to evaluate on from a CSV file.

    The CSV must contain a `relative_path` column with values in the format
    'exercise/person/session' (backslash or forward slash separators accepted).
    All other columns are ignored.

    Returns
    -------
    set of (exercise, person, session) tuples — the recordings that should
    be included in evaluation. Any recording not in this set should be
    skipped during scoring (but may still be used for training).
    """
    import csv as _csv

    eval_set = set()
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            raw_path = row['relative_path'].replace('\\', '/')
            parts    = [p for p in raw_path.split('/') if p]
            if len(parts) >= 3:
                eval_set.add(tuple(parts[-3:]))

    return eval_set


def match_recording_to_eval_set(rec_dir, eval_set):
    """
    Return True if the recording directory's last three path components
    match any entry in eval_set.

    Parameters
    ----------
    rec_dir  : path-like recording directory
    eval_set : set of (exercise, person, session) tuples from load_eval_sessions
    """
    parts = [p for p in Path(rec_dir).parts if p]
    if len(parts) < 3:
        return False
    return tuple(parts[-3:]) in eval_set


def match_recording_to_boundaries(rec_dir, boundaries):
    """
    Find the boundary list for a recording directory by matching the last
    three path components against the keys in the boundaries dict.

    Parameters
    ----------
    rec_dir    : path-like — the recording directory
    boundaries : dict returned by load_rep_boundaries()

    Returns
    -------
    list of (start_s, end_s) pairs, or None if no match found.
    """
    parts = [p for p in Path(rec_dir).parts if p]
    if len(parts) < 3:
        return None
    key = tuple(parts[-3:])
    return boundaries.get(key, None)


# ---------------------------------------------------------------------------
# Window utilities
# ---------------------------------------------------------------------------

def compute_sync_window(signals, trim_margin_s=2.0):
    """
    Compute the overlapping time window across all signals, inset by
    trim_margin_s on each end to remove startup/shutdown noise.

    Parameters
    ----------
    signals        : dict of signal dicts, each containing a 'ts' array
    trim_margin_s  : seconds to trim from each end (default 2.0)

    Returns
    -------
    (t_start, t_end) in Unix epoch seconds, or raises ValueError if the
    trimmed window is empty.
    """
    raw_start = max(d['ts'][0]  for d in signals.values())
    raw_end   = min(d['ts'][-1] for d in signals.values())

    t_start = raw_start + trim_margin_s
    t_end   = raw_end   - trim_margin_s

    if t_start >= t_end:
        raise ValueError(
            f"No valid time window after trimming {trim_margin_s}s from each end.\n"
            f"  Raw overlap: {raw_start:.3f} -> {raw_end:.3f} "
            f"({raw_end - raw_start:.2f}s)\n"
            f"  After trim:  {t_start:.3f} -> {t_end:.3f}")

    return t_start, t_end


def trim_to_window(ts, xs, ys, zs, t_start, t_end):
    """Keep only samples within [t_start, t_end]."""
    mask = (ts >= t_start) & (ts <= t_end)
    return ts[mask], xs[mask], ys[mask], zs[mask]


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def compute_fs(times):
    """Estimate sampling rate from median inter-sample interval."""
    dt = float(np.median(np.diff(times)))
    return 1.0 / dt if dt > 0 else 100.0


def lowpass_filter(signal, fs, cutoff_hz=5.0):
    """2nd-order Butterworth low-pass filter."""
    nyq    = fs / 2.0
    cutoff = min(cutoff_hz, nyq * 0.9)
    b, a   = butter(2, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)


def bandpass_filter(mag, fs, period_s, lower_cycles=0.5, upper_cycles=2.5):
    """
    Adaptive bandpass filter with cutoffs derived from the estimated rep period.

    Cutoffs:
      low  = lower_cycles / period_s Hz  (default 0.5x rep frequency)
             Removes DC offset and slow sensor drift.
      high = upper_cycles / period_s Hz  (default 2.5x rep frequency)
             Preserves the fundamental and first two harmonics.

    Both cutoffs are clamped before use. Returns (mag_bp, low_hz, high_hz).
    Falls back to returning a copy of mag if the range is degenerate.
    """
    nyq     = fs / 2.0
    low_hz  = max(lower_cycles / period_s, 0.01)
    high_hz = min(upper_cycles / period_s, nyq * 0.95)

    if low_hz >= high_hz:
        return mag.copy(), low_hz, high_hz

    b, a   = butter(2, [low_hz / nyq, high_hz / nyq], btype='band')
    mag_bp = filtfilt(b, a, mag)
    return mag_bp, low_hz, high_hz


def estimate_rep_period_acf(mag_f, fs, min_period_s=0.3, max_period_s=10.0):
    """
    Estimate the dominant rep period in seconds via normalised ACF.

    Uses FFT-based ACF (O(n log n)). Searches for the highest-prominence
    peak between min_period_s and max_period_s. Falls back to
    max_period_s / 2 if no clear peak is found.
    """
    x     = mag_f - mag_f.mean()
    n     = len(x)
    fft_x = np.fft.rfft(x, n=2 * n)
    acf   = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf   = acf / (acf[0] + 1e-12)

    min_lag = max(1, int(min_period_s * fs))
    max_lag = min(n - 1, int(max_period_s * fs))

    if min_lag >= max_lag:
        fallback = max_period_s / 2.0
        print(f"    ACF: search range empty, falling back to {fallback:.2f}s")
        return fallback

    peaks, props = find_peaks(acf[min_lag:max_lag], prominence=0.05)
    if len(peaks) == 0:
        fallback = max_period_s / 2.0
        print(f"    ACF: no dominant peak found, falling back to {fallback:.2f}s")
        return fallback

    best     = peaks[np.argmax(props['prominences'])]
    lag_idx  = best + min_lag
    period_s = lag_idx / fs
    print(f"    ACF: estimated rep period = {period_s:.2f}s "
          f"(lag {lag_idx}, prominence "
          f"{props['prominences'][np.argmax(props['prominences'])]:.3f})")
    return period_s


# ---------------------------------------------------------------------------
# Peak / valley detection
# ---------------------------------------------------------------------------

def detect_peaks_valleys(times, xs, ys, zs,
                          lowpass_hz=5.0,
                          min_separation_s=None,
                          prominence_factor=0.5):
    """
    Full peak and valley detection used by plot_multi_accel.py.

    Pipeline:
      1. Vector magnitude
      2. Low-pass  → mag_f  (retained for DTW scoring)
      3. ACF period estimate
      4. Bandpass  → mag_bp  (used for detection only)
      5. find_peaks on mag_bp (peaks) and -mag_bp (valleys)

    Returns
    -------
    peak_times, valley_times, valley_idx, mag_f, mag_bp,
    bp_low_hz, bp_high_hz, min_sep_used
    """
    dt = float(np.median(np.diff(times)))
    fs = 1.0 / dt if dt > 0 else 100.0

    mag   = np.sqrt(xs**2 + ys**2 + zs**2)
    nyq   = fs / 2.0
    cut   = min(lowpass_hz, nyq * 0.9)
    b, a  = butter(2, cut / nyq, btype='low')
    mag_f = filtfilt(b, a, mag)

    if min_separation_s is None:
        period_s     = estimate_rep_period_acf(mag_f, fs)
        min_sep_used = period_s / 2.0
    else:
        period_s     = min_separation_s * 2.0
        min_sep_used = min_separation_s

    mag_bp, bp_low_hz, bp_high_hz = bandpass_filter(mag_f, fs, period_s)

    dist    = max(1, int(min_sep_used * fs))
    iqr     = float(np.percentile(mag_bp, 75) - np.percentile(mag_bp, 25))
    prom    = max(iqr * prominence_factor, 1e-6)

    peak_idx,   _ = find_peaks( mag_bp, distance=dist, prominence=prom)
    valley_idx, _ = find_peaks(-mag_bp, distance=dist, prominence=prom)

    return (np.array(times)[peak_idx],
            np.array(times)[valley_idx],
            valley_idx,
            mag_f,
            mag_bp,
            bp_low_hz,
            bp_high_hz,
            min_sep_used)


def detect_valleys(times, xs, ys, zs,
                   lowpass_hz=5.0,
                   min_separation_s=None,
                   prominence_factor=0.5):
    """
    Simplified valley-only detection used by classify_exercise.py and
    autoencoder_form.py.

    Same two-stage pipeline as detect_peaks_valleys but returns only
    (valley_idx, mag_f, min_sep_used).  mag_f (low-pass) is returned
    for downstream feature extraction; valley detection runs on mag_bp.
    """
    fs    = compute_fs(times)
    mag   = np.sqrt(xs**2 + ys**2 + zs**2)
    mag_f = lowpass_filter(mag, fs, lowpass_hz)

    if min_separation_s is None:
        period_s         = estimate_rep_period_acf(mag_f, fs)
        min_separation_s = period_s / 2.0
    else:
        period_s = min_separation_s * 2.0

    mag_bp, _, _ = bandpass_filter(mag_f, fs, period_s)

    dist    = max(1, int(min_separation_s * fs))
    iqr     = float(np.percentile(mag_bp, 75) - np.percentile(mag_bp, 25))
    prom    = max(iqr * prominence_factor, 1e-6)
    vidx, _ = find_peaks(-mag_bp, distance=dist, prominence=prom)

    return vidx, mag_f, min_separation_s


# ---------------------------------------------------------------------------
# Rep segmentation
# ---------------------------------------------------------------------------

def segment_reps(valley_idx, n_samples=None, min_dur_factor=0.5, n_template=5):
    """
    Slice the signal into valley-to-valley rep segments.

    Estimates a typical rep length from the first n_template inter-valley
    gaps and rejects segments shorter than min_dur_factor × that median.
    This guards against spurious mid-rep valleys splitting a rep in two.

    Parameters
    ----------
    valley_idx     : 1-D array of sample indices of detected valleys
    n_samples      : total signal length (unused, kept for API compatibility)
    min_dur_factor : minimum rep length as a fraction of median rep length
    n_template     : how many early reps to use for the duration estimate

    Returns
    -------
    list of (start_idx, end_idx) tuples
    """
    if len(valley_idx) < 2:
        return []

    early_durs = np.diff(valley_idx[:n_template + 1])
    median_dur = float(np.median(early_durs))
    min_dur    = max(1, int(median_dur * min_dur_factor))

    return [
        (int(valley_idx[i]), int(valley_idx[i + 1]))
        for i in range(len(valley_idx) - 1)
        if valley_idx[i + 1] - valley_idx[i] >= min_dur
    ]
